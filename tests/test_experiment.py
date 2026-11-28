import json
import math

import numpy as np
import pytest
from problems import neg_sphere_obj, sphere_obj

import ypea
from ypea.experiment import Experiment, Report, stats, tables


def make_problems():
    return [
        ypea.Problem(ypea.Real("x", shape=3, lb=-5, ub=5), sphere_obj, name="sphere3", optimum=0.0),
        ypea.Problem(ypea.Real("x", shape=2, lb=-5, ub=5), neg_sphere_obj, "max", name="negsphere2"),
    ]


def make_exp(out, **kw):
    algs = {"PSO": ypea.PSO(pop_size=10), "DE": ypea.DE(pop_size=10)}
    return Experiment(algs, make_problems(), **{"runs": 3, "max_nfe": 200, "seed": 7, "output": out, **kw})


def strip_times(rec):
    rec = json.loads(json.dumps(rec))
    del rec["result"]["run_time"], rec["result"]["history"]["time"]
    return rec


def run_files(out):
    return {p.relative_to(out).as_posix(): strip_times(json.loads(p.read_text("utf-8")))
            for p in sorted((out / "runs").rglob("run_*.json"))}


def test_runs_resume_and_metadata(tmp_path):
    exp = make_exp(tmp_path).run()
    files = sorted((tmp_path / "runs").rglob("run_*.json"))
    assert len(files) == 2 * 2 * 3
    rec = json.loads((tmp_path / "runs/PSO/sphere3/run_0.json").read_text("utf-8"))
    assert rec["result"]["nfe"] == 200 and rec["sense"] == "min"
    assert rec["checkpoints"]["nfe"][0] == 2 and rec["checkpoints"]["nfe"][-1] == 200
    assert rec["checkpoints"]["error"][-1] == rec["result"]["error"]
    assert rec["checkpoints"]["error"] == sorted(rec["checkpoints"]["error"], reverse=True)
    meta = json.loads((tmp_path / "metadata.json").read_text("utf-8"))
    assert meta["versions"]["ypea"] == ypea.__version__ and meta["runs"] == 3
    # the caller's algorithms keep their own limits
    assert exp.algorithms["PSO"].max_nfe is None

    stamps = {f: f.stat().st_mtime_ns for f in files}
    make_exp(tmp_path).run()
    assert {f: f.stat().st_mtime_ns for f in files} == stamps  # nothing was run again

    # a changed setting changes the key, so only those runs are made again
    changed = make_exp(tmp_path, max_nfe=300)
    old_keys = {t["key"] for t in make_exp(tmp_path)._tasks()}
    assert not old_keys & {t["key"] for t in changed._tasks()}
    other = Experiment({"PSO": ypea.PSO(pop_size=12), "DE": ypea.DE(pop_size=10)}, make_problems(),
                       runs=3, max_nfe=200, seed=7, output=tmp_path)
    other.run()
    assert {f for f in files if f.stat().st_mtime_ns != stamps[f]} == set((tmp_path / "runs/PSO").rglob("*.json"))
    # an old record no longer counts once the settings changed
    assert len(Report.load(tmp_path).records("PSO", "sphere3")) == 3


def test_n_jobs_gives_identical_files(tmp_path):
    make_exp(tmp_path / "a", n_jobs=1).run()
    make_exp(tmp_path / "b", n_jobs=2).run()
    assert run_files(tmp_path / "a") == run_files(tmp_path / "b")


def first_value(out, label, problem):
    rec = json.loads((out / f"runs/{label}/{problem}/run_0.json").read_text("utf-8"))
    return rec["result"]["trace"]["obj_value"][0]


def test_seeds_do_not_depend_on_other_pairs(tmp_path):
    make_exp(tmp_path / "a").run()
    # numpy rule: independent streams (PSO and DE draw their first population the same way)
    assert first_value(tmp_path / "a", "PSO", "sphere3") != first_value(tmp_path / "a", "DE", "sphere3")
    p = make_problems()[1]
    Experiment({"DE": ypea.DE(pop_size=10)}, [p], runs=3, max_nfe=200, seed=7, output=tmp_path / "b").run()
    a, b = run_files(tmp_path / "a"), run_files(tmp_path / "b")
    assert all(b[k] == a[k] for k in b) and len(b) == 3


def test_studio_seed_rule(tmp_path):
    exp = make_exp(tmp_path, seeds="studio").run()
    rec = json.loads((tmp_path / "runs/DE/sphere3/run_2.json").read_text("utf-8"))
    assert rec["result"]["seed"] == {"studio": ypea.derive_seed(7, 2)}
    alone = ypea.DE(pop_size=10, max_nfe=200).solve(make_problems()[0], seed=ypea.StudioRng(ypea.derive_seed(7, 2)))
    assert rec["result"]["best"]["obj_value"] == alone.best.obj_value
    # common random numbers: the same first population for both algorithms
    assert first_value(tmp_path, "PSO", "sphere3") == first_value(tmp_path, "DE", "sphere3")
    assert exp.report().values("DE", "sphere3").shape == (3,)
    with pytest.raises(ValueError):
        make_exp(tmp_path, seeds="studio", seed=-1)


def test_in_memory_and_list_labels():
    exp = Experiment([ypea.DE(pop_size=10), ypea.DE(pop_size=12)], make_problems()[:1], runs=2, max_nfe=100)
    assert list(exp.algorithms) == ["DE", "DE (2)"]
    rep = exp.run().report()
    assert rep.values("DE (2)", "sphere3").shape == (2,)
    with pytest.raises(ValueError):
        Experiment([ypea.DE()], [make_problems()[0]] * 2)


def test_sweep():
    grid = {"pop_size": [10], "crossover_prob": [0.5, 0.8]}
    exp = Experiment.sweep(ypea.DE, grid, make_problems()[:1], runs=1, max_nfe=50)
    assert list(exp.algorithms) == ["DE(crossover_prob=0.5)", "DE(crossover_prob=0.8)"]
    alg = exp._algs["DE(crossover_prob=0.8)"]
    assert alg.crossover_prob == 0.8 and alg.pop_size == 10 and alg.max_nfe == 50
    exp = Experiment.sweep(ypea.DE, [{"beta_min": 0.3}, {"beta_min": 0.4, "crossover_prob": 0.1}], make_problems()[:1])
    assert list(exp.algorithms) == ["DE(beta_min=0.3)", "DE(beta_min=0.4, crossover_prob=0.1)"]


def test_exports_contain_the_numbers(tmp_path):
    pytest.importorskip("scipy", reason="the rank sum test needs pip install ypea[stats]")
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    make_exp(tmp_path).run()
    rep = Report.load(tmp_path)
    t = rep.table(baseline="DE")
    err = rep.values("PSO", "sphere3")
    assert np.allclose(err, [json.loads((tmp_path / f"runs/PSO/sphere3/run_{i}.json").read_text("utf-8"))
                             ["result"]["error"] for i in range(3)])
    mean = format(np.mean(err), ".3e")
    obj = format(np.median(rep.values("DE", "negsphere2")), ".3e")  # no optimum: objective values
    for text in (t.to_csv(tmp_path / "t.csv"), t.to_markdown(tmp_path / "t.md"), t.to_latex(tmp_path / "t.tex"),
                 t.to_html()):
        assert mean in text and obj in text
    tex = (tmp_path / "t.tex").read_text("utf-8")
    assert "−" not in tex and "\\toprule" in tex and "\\textbf{" in tex and "\\bottomrule" in tex
    assert "**" in t.to_markdown() and "vs DE" in t.to_csv()
    page = rep.save_html().read_text("utf-8")
    assert mean in page and "data:image/png;base64," in page and "PSO" in page


def test_partial_folder_and_dataframe(tmp_path):
    pytest.importorskip("scipy", reason="the rank sum test needs pip install ypea[stats]")
    make_exp(tmp_path).run()
    for f in (tmp_path / "runs/PSO/negsphere2").glob("*.json"):
        f.unlink()  # as if the experiment was interrupted
    rep = Report.load(tmp_path)
    t = rep.table(baseline="DE")
    assert ("negsphere2", "PSO") not in t.data["marks"] and ("sphere3", "PSO") in t.data["marks"]
    assert "n/a" in t.to_markdown()
    pytest.importorskip("pandas")
    df = t.to_dataframe()
    assert list(df.columns) == ["Problem", "Statistic", "PSO", "DE"] and len(df) == len(t.rows)


def test_table_marks_bold_and_ranks():
    pytest.importorskip("scipy", reason="the rank sum test needs pip install ypea[stats]")
    rng = np.random.default_rng(0)
    samples = {"f1": {"A": rng.random(10), "B": rng.random(10) + 5, "C": rng.random(10)},
               "f2": {"A": rng.random(10), "B": rng.random(10) + 5, "C": np.full(10, 9.0)},
               "f3": {"A": rng.random(10) - 5, "B": rng.random(10), "C": rng.random(10) + 5}}
    smaller = {"f1": True, "f2": True, "f3": False}  # f3 is a maximization
    t = tables.comparison_table(samples, ["A", "B", "C"], list(samples), smaller, baseline="A")
    assert t.data["marks"] == {("f1", "B"): "-", ("f1", "C"): "=", ("f2", "B"): "-", ("f2", "C"): "-",
                               ("f3", "B"): "+", ("f3", "C"): "+"}
    assert t.rows[-3][3]["text"] == "1/0/2" and t.rows[-3][4]["text"] == "1/1/1"
    assert "−" in t.to_markdown() and "$-$" in t.to_latex()
    mean_f3 = t.rows[10]
    assert mean_f3[1]["text"] == "Mean" and mean_f3[4]["bold"] and not mean_f3[2]["bold"]
    expected = stats.friedman([[np.mean(samples[p][a]) * (1 if smaller[p] else -1) for a in "ABC"]
                               for p in samples])["ranks"]
    assert np.allclose(t.data["friedman"]["ranks"], expected)


def test_tables_without_scipy(monkeypatch):
    monkeypatch.setattr(stats, "scipy_stats", lambda: None)
    samples = {"f": {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}}
    t = tables.comparison_table(samples, ["A", "B"], ["f"], {"f": True})
    assert "1.500e+00" in t.to_markdown() and "pip install ypea[stats]" in t.to_markdown()
    assert not t.data["marks"]
    with pytest.raises(ImportError, match="ypea\\[stats\\]"):
        stats.rank_sum([1, 2], [3, 4])


def test_rank_sum_matches_scipy():
    st = pytest.importorskip("scipy.stats", reason="the rank sum test needs pip install ypea[stats]")
    a = [0.11, 0.25, 0.31, 0.42, 0.18, 0.27, 0.33]
    b = [0.52, 0.47, 0.61, 0.38, 0.55, 0.49, 0.58]
    mark, p = stats.rank_sum(a, b)
    assert mark == "+" and p == st.mannwhitneyu(a, b, alternative="two-sided").pvalue
    assert stats.rank_sum(b, a)[0] == "-" and stats.rank_sum(a, b, smaller_is_better=False)[0] == "-"
    assert stats.rank_sum([1, 1, 1], [1, 1, 1]) == ("=", 1.0)


def test_friedman_by_hand_and_against_scipy():
    # 4 problems, 3 algorithms. Ranks per problem: (1,2,3), (1,3,2), (2,1,3), (1,2.5,2.5)
    v = [[1, 2, 3], [1, 3, 2], [5, 4, 6], [0, 7, 7]]
    fr = stats.friedman(v)
    assert np.allclose(fr["ranks"], [5 / 4, 8.5 / 4, 10.5 / 4])
    # 12n/(k(k+1)) (sum R^2 - k(k+1)^2/4) = 4 (12.96875 - 12) = 3.875, tie correction 1 - 6/96
    assert fr["chi2"] == pytest.approx(3.875 / (1 - 6 / 96))
    st = pytest.importorskip("scipy.stats", reason="p values need pip install ypea[stats]")
    ref = st.friedmanchisquare(*np.array(v, dtype=float).T)
    assert fr["chi2"] == pytest.approx(ref.statistic) and fr["p"] == pytest.approx(ref.pvalue)
    assert fr["f"] == pytest.approx(3 * fr["chi2"] / (4 * 2 - fr["chi2"]))
    # larger is better reverses the ranks
    assert np.allclose(stats.friedman(v, smaller_is_better=False)["ranks"], 4 - fr["ranks"])


def test_posthoc_matches_derrac_example():
    # Derrac et al. (2011), Example 6, Tables 12 and 13: k = 4, n = 25, SaDE is the control
    ranks = [2.48, 3.12, 2.44, 1.96]  # IPOP-CMA-ES, CHC, SS-BLX, SaDE
    ph = stats.posthoc(ranks, 25, control=3)
    assert [round(ph["z"][i], 6) for i in (1, 0, 2)] == [3.176791, 1.424079, 1.314534]
    assert [round(ph["p"][i], 6) for i in (1, 0, 2)] == [0.001489, 0.154424, 0.188667]
    # Holm: 3 p1, then max(previous, 2 p2), then max(previous, p3)
    assert ph["holm"][1] == pytest.approx(3 * ph["p"][1])
    assert ph["holm"][0] == pytest.approx(max(3 * ph["p"][1], 2 * ph["p"][0]))
    assert ph["holm"][2] == pytest.approx(max(ph["holm"][0], ph["p"][2]))
    assert math.isclose(stats.rankdata([3, 1, 3, 2]).sum(), 10)


def test_zero_below_follows_the_cec_rule(tmp_path):
    import ypea
    from ypea.experiment import Experiment

    p = ypea.benchmarks.problem("sphere", 3)
    exp = Experiment({"NM": ypea.NelderMead()}, [p], runs=2, max_nfe=3000, output=tmp_path / "e")
    exp.run()
    rep = exp.report()
    raw = rep.values("NM", p.name)
    rep.zero_below = 1e-8
    floored = rep.values("NM", p.name)
    assert np.all(raw < 1e-8) and np.all(floored == 0)


# several objectives


def mo_problems():
    return [ypea.benchmarks.multi_problem("zdt1", dim=6, front_points=200),
            ypea.benchmarks.multi_problem("dtlz2", m=3, front_points=100)]


def test_igd_and_hypervolume_are_stored_and_compared(tmp_path):
    algs = {"NSGA-II": ypea.NSGAII(pop_size=20), "RS": ypea.RandomSearch(pop_size=20)}
    exp = Experiment(algs, mo_problems(), runs=3, max_nfe=600, seed=1, output=tmp_path, checkpoints=(0.5, 1.0))
    rep = exp.run().report()
    meta = rep.metadata["problems"][0]["indicators"]
    assert meta["front_points"] == 200 and meta["ideal"] == [0.0, 0.0] and meta["ref_point"] == [1.1, 1.1]
    p = mo_problems()[1]
    rec = rep.records("RS", p.name)[0]
    front = np.array(rec["result"]["front"]["obj_values"])
    igd = ypea.indicators.igd(front, p.info["pareto_front"])
    assert rec["indicators"]["igd"] == pytest.approx(igd)
    # the DTLZ2 front spans [0, 1] in every objective, so the normalized values are the raw ones
    assert rec["indicators"]["hypervolume"] == pytest.approx(ypea.indicators.hypervolume(front, [1.1] * 3))
    cp = rec["checkpoints"]
    assert cp["nfe"] == [300, 600] and cp["igd"][1] == pytest.approx(igd) and cp["igd"][0] >= cp["igd"][1]
    assert cp["hypervolume"][0] <= cp["hypervolume"][1]
    # error means IGD with a true front, smaller is better, and hypervolume is larger is better
    t = rep.table(stats=("mean",))
    # without SciPy the last note says the rank sum test is missing
    assert ("mean IGD" in t.notes[-1] or stats.scipy_stats() is None) and rep.smaller_is_better(p.name)
    assert t.data["describe"][p.name]["RS"]["mean"] == pytest.approx(rep.values("RS", p.name, "igd").mean())
    assert not rep.smaller_is_better(p.name, "hypervolume")
    t = rep.table(metric="hypervolume", stats=("mean",))
    assert "mean hypervolume" in t.notes[-1] or stats.scipy_stats() is None
    assert t.data["friedman"]["ranks"].tolist() == [1.0, 2.0]  # NSGA-II beats random search
    grid, curves = rep.curves("RS", p.name, "igd")
    assert grid.tolist() == [300, 600] and curves[0].tolist() == pytest.approx(cp["igd"])
    with pytest.raises(ValueError, match="igd"):
        rep.values("RS", p.name, "obj_value")
    with pytest.raises(ValueError, match="one objective"):
        Report.load(make_exp(tmp_path / "single").run().output).values("PSO", "sphere3", "igd")


def test_several_objectives_follow_the_senses_and_keep_single_objective_keys(tmp_path):
    def run(problems, out):
        exp = Experiment([ypea.RandomSearch(pop_size=20)], problems, runs=1, max_nfe=200, seed=2, output=out)
        return exp.run().report()

    p = mo_problems()[0]
    q = ypea.Problem(p.vars, Negated(p.objective), vectorized=True, sense="max", n_obj=2, name=p.name,
                     info={"pareto_front": -p.info["pareto_front"]})
    a, b = (run([x], tmp_path / str(i)).records("RS", p.name)[0] for i, x in enumerate((p, q)))
    assert a["indicators"] == b["indicators"] and a["checkpoints"]["hypervolume"] == b["checkpoints"]["hypervolume"]
    # mixed senses: only the second objective is maximized
    flip = np.array([1.0, -1.0])
    mixed = ypea.Problem(p.vars, lambda sol: p.objective(sol) * flip, vectorized=True, sense=["min", "max"],
                         n_obj=2, name=p.name, info={"pareto_front": p.info["pareto_front"] * flip})
    c = run([mixed], tmp_path / "mixed").records("RS", p.name)[0]
    assert a["indicators"] == c["indicators"] and a["checkpoints"]["igd"] == c["checkpoints"]["igd"]
    # single objective runs have no indicators, and their keys do not mention them
    rec = run(make_problems()[:1], tmp_path / "s").records("RS", "sphere3")[0]
    assert "indicators" not in rec and "indicators" not in rec["settings"]
    # without max_nfe there are no checkpoints, but the final values are there
    exp = Experiment([ypea.RandomSearch(pop_size=20, max_iter=3)], [p], runs=2, seed=2)
    rep = exp.run().report()
    assert rep.records("RS", p.name)[0]["checkpoints"] is None and len(rep.values("RS", p.name)) == 2
    with pytest.raises(ValueError, match="checkpoints"):
        rep.curves("RS", p.name)


class Negated:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, sol):
        return -self.fn(sol)


def test_multi_objective_report_figures_and_html(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp = Experiment([ypea.RandomSearch(pop_size=20), ypea.MOEAD(pop_size=20)], mo_problems()[:1], runs=2,
                     max_nfe=300, seed=0, output=tmp_path)
    rep = exp.run().report()
    ax = rep.plot_convergence(mo_problems()[0].name)
    assert ax.get_ylabel() == "IGD"
    plt.close("all")
    page = rep.save_html(metric="hypervolume").read_text("utf-8")
    assert "mean hypervolume" in page and page.count("<img") == 3
