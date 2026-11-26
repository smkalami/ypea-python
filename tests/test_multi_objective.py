"""Several objectives: dominance, sorting, crowding, the archive, results and scalarize."""

import pickle

import numpy as np
import pytest

import ypea
from ypea.comparison import Comparator
from ypea.evaluation import Evaluator

# small test problems


def schaffer(sol):
    x = sol.x[0]
    return [x**2, (x - 2) ** 2]


def two_obj(sol):
    x = sol.x
    return np.array([x[0], (1 + x[1]) * (1 - np.sqrt(x[0])) + 0.1 * x[2]])


def two_obj_vec(sol):
    x = sol.x
    return np.column_stack([x[:, 0], (1 + x[:, 1]) * (1 - np.sqrt(x[:, 0])) + 0.1 * x[:, 2]])


def rounded_constrained(sol):
    # rounded values give exact ties and duplicates, some rows are NaN, many are infeasible
    x = sol.x
    f = [float(np.round(x[0], 1)), float(np.round(1 - x[0] + x[1] ** 2, 1))]
    if x[2] > 0.95:
        f[1] = float("nan")
    return f, float(np.round(0.3 - x[1], 1))


def problem3(sense="min", constraints="feasibility"):
    return ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), rounded_constrained, sense=sense, n_obj=2,
                        constraints=constraints, penalty=2.0)


# plain loop references


def dominates_ref(f1, v1, f2, v2, sign):
    if v1 != v2:
        return v1 < v2
    a, b = [s * x for s, x in zip(sign, f1, strict=True)], [s * x for s, x in zip(sign, f2, strict=True)]
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(x < y for x, y in zip(a, b, strict=True))


def archive_ref(F, V, sign):
    """Rows that no other row dominates, and that do not repeat an earlier row."""
    out = []
    for i in range(len(V)):
        if any(dominates_ref(F[j], V[j], F[i], V[i], sign) for j in range(len(V))):
            continue
        if any(list(F[j]) == list(F[i]) and V[j] == V[i] for j in range(i)):
            continue
        out.append((*F[i], V[i]))
    return sorted(out)


def rows(F, V):
    return sorted((*f, v) for f, v in zip(F.tolist(), V.tolist(), strict=True))


# Problem


def test_problem_senses_and_errors():
    p = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), schaffer, n_obj=2)
    assert p.n_obj == 2 and p.sense == ["min", "min"]
    p = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), schaffer, sense=["min", "maximize"])
    assert p.n_obj == 2 and p.sense == ["min", "max"] and p.comparator.sign.tolist() == [1, -1]
    q = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), lambda s: 1.0, sense=["max"])
    assert q.n_obj == 1 and q.sense == "max"
    with pytest.raises(ValueError, match="entries"):
        ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), schaffer, sense=["min", "max"], n_obj=3)
    with pytest.raises(ValueError, match="n_obj"):
        ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), schaffer, n_obj=0)


def test_objective_value_errors():
    p = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), lambda s: (1.0, 2.0), n_obj=2)
    with pytest.raises(ValueError, match="tuple"):
        ypea.RandomSearch(max_nfe=10).solve(p)  # (f1, f2) as a tuple reads as (f, g)
    p = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), lambda s: [1.0, 2.0, 3.0], n_obj=2)
    with pytest.raises(ValueError, match="2 values"):
        ypea.RandomSearch(max_nfe=10).solve(p)
    p = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), lambda s: s.x[:, :2].T, n_obj=2, vectorized=True)
    with pytest.raises(ValueError, match="shape"):
        ypea.RandomSearch(max_nfe=10).solve(p)


# dominance, sorting and crowding


def test_dominance_cases():
    c = Comparator(["min", "min"])
    d = lambda f1, v1, f2, v2: bool(c.dominates(np.array(f1, float), v1, np.array(f2, float), v2))  # noqa: E731
    assert d([1, 1], 0, [2, 2], 0) and d([1, 2], 0, [1, 3], 0)
    assert not d([1, 1], 0, [1, 1], 0)  # equal
    assert not d([1, 3], 0, [2, 2], 0) and not d([2, 2], 0, [1, 3], 0)  # incomparable
    assert d([9, 9], 0, [0, 0], 0.1)  # feasible beats infeasible
    assert d([9, 9], 0.1, [0, 0], 0.2) and not d([0, 0], 0.2, [9, 9], 0.1)  # smaller violation
    assert d([1, 1], 0.2, [2, 2], 0.2)  # equal violation: Pareto
    assert not d([1, 1], np.inf, [2, 2], 0.5)
    mx = Comparator(["max", "min"])
    assert mx.dominates(np.array([2.0, 1.0]), 0, np.array([1.0, 1.0]), 0)
    assert not mx.dominates(np.array([1.0, 1.0]), 0, np.array([2.0, 1.0]), 0)
    pen = Comparator(["min", "min"], "penalty", 10.0)
    assert pen.dominates(np.array([0.0, 0.0]), 0.05, np.array([1.0, 1.0]), 0)  # 0.5 < 1
    assert not pen.dominates(np.array([0.0, 0.0]), 0.2, np.array([1.0, 1.0]), 0)
    assert pen.dominates(np.array([1.0, 1.0]), 0, np.array([0.0, 0.0]), 0.1)  # equal penalized, less violation
    assert c.better(np.array([1.0, 1.0]), 0, np.array([2.0, 2.0]), 0)
    with pytest.raises(ValueError, match="single objective"):
        c.argsort(np.zeros((3, 2)), np.zeros(3))


@pytest.mark.parametrize("constraints", ["feasibility", "penalty"])
def test_one_objective_dominance_matches_better(constraints):
    c = Comparator("max", constraints, 3.0)
    rng = np.random.default_rng(0)
    F, V = rng.integers(0, 4, (60, 1)).astype(float), rng.integers(0, 3, 60) * 0.5
    a = c.dominates(F[:, None], V[:, None], F[None], V[None])
    b = c.better(F[:, None], V[:, None], F[None], V[None])
    assert (a == b).all()


def brute_sort(F, V, sign):
    left, ranks, r = set(range(len(V))), np.full(len(V), -1), 0
    while left:
        front = [i for i in left if not any(dominates_ref(F[j], V[j], F[i], V[i], sign) for j in left)]
        ranks[front] = r
        left -= set(front)
        r += 1
    return ranks


@pytest.mark.parametrize("seed", range(6))
def test_nondominated_sort_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    m = 2 + seed % 2
    sense = ["min", "max", "min"][:m]
    F = rng.integers(0, 5, (50, m)).astype(float)  # many ties and duplicates
    V = np.where(rng.random(50) < 0.3, rng.integers(1, 3, 50) * 0.5, 0.0)
    c = Comparator(sense)
    assert c.nondominated_sort(F, V).tolist() == brute_sort(F, V, c.sign).tolist()


def test_crowding_distance_hand_cases():
    c = Comparator(["min", "min"])
    F = np.array([[0, 4], [1, 3], [3, 1], [4, 0]], float)
    d = c.crowding_distance(F)
    # middle points: (3 - 0) / 4 + (4 - 1) / 4 and (4 - 1) / 4 + (3 - 0) / 4
    assert d[0] == d[3] == np.inf and d[1] == pytest.approx(1.5) and d[2] == pytest.approx(1.5)
    assert np.all(c.crowding_distance(F[:2]) == np.inf)
    d = c.crowding_distance(np.array([[0, 2], [1, 2], [2, 2]], float))  # one objective has no range
    assert d.tolist() == [np.inf, 1.0, np.inf]
    # per front: ranks separate the two fronts
    G = np.vstack([F, F + 10])
    d = c.crowding_distance(G, ranks=np.array([0] * 4 + [1] * 4))
    assert d[:4].tolist() == d[4:].tolist()
    # scale does not matter (normalized)
    assert c.crowding_distance(F * [1, 1000]).tolist() == pytest.approx(c.crowding_distance(F).tolist())


# the archive


@pytest.mark.parametrize("sense", ["min", ["max", "min"]])
@pytest.mark.parametrize("constraints", ["feasibility", "penalty"])
@pytest.mark.parametrize("max_nfe", [None, 300])
def test_archive_matches_brute_force(sense, constraints, max_nfe):
    p = problem3(sense, constraints)
    rng = np.random.default_rng(1)
    batches = [rng.random((n, 3)) for n in (1, 7, 50, 3, 200, 60)]
    ev = Evaluator(p, max_nfe=max_nfe, keep_evaluations=True)
    for X in batches:
        out = ev(X)
        assert len(out) == len(X)
    X = np.concatenate(batches)[: max_nfe or None]
    F, V = [], []
    for x in X:
        f, v = p.evaluate(p.decode_one(x))
        F.append(f if np.all(np.isfinite(f)) else p.comparator.worst_value)
        V.append(v)
    F, V = np.array(F), np.array(V)
    sign = p.comparator.sign
    assert rows(ev.archive.F, ev.archive.V) == archive_ref(F, V, sign)
    # the trace rebuilds the archive at every NFE
    tn, tr = np.array(ev.trace_nfe), np.array(ev.trace_removed)
    tf, tv = np.array(ev.trace_f), np.array(ev.trace_v)
    for t in (1, 5, 30, 61, 200, len(V)):
        alive = (tn <= t) & ((tr < 0) | (tr > t))
        assert rows(tf[alive], tv[alive]) == archive_ref(F[:t], V[:t], sign), t


def test_archive_keeps_least_violating_until_feasible():
    p = ypea.Problem(ypea.Real("x", shape=2, lb=0, ub=1), lambda s: ([s.x[0], 1 - s.x[0]], 0.5 - s.x[1]), n_obj=2)
    ev = Evaluator(p)
    ev(np.array([[0.1, 0.1], [0.2, 0.3], [0.9, 0.3]]))  # all infeasible, the last two equally
    assert sorted(ev.archive.V.tolist()) == pytest.approx([0.2, 0.2])
    ev(np.array([[0.5, 0.9]]))  # feasible: replaces them all
    assert ev.archive.V.tolist() == [0.0] and len(ev.archive) == 1


def test_budget_cut_rows_never_enter():
    p = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), two_obj, n_obj=2)
    ev = Evaluator(p, max_nfe=5)
    out = ev(np.random.default_rng(0).random((8, 3)))
    assert np.all(out.F[5:] == np.inf) and np.all(out.V[5:] == np.inf)
    assert ev.nfe == 5 and ev.stop_reason == "max_nfe" and np.all(np.isfinite(ev.archive.F))
    assert max(ev.trace_nfe) <= 5


def test_max_archive_truncates_by_crowding():
    p = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), two_obj, n_obj=2)
    ev = Evaluator(p, max_archive=10)
    for X in np.random.default_rng(2).random((20, 50, 3)):
        ev(X)
        assert len(ev.archive) <= 10
    ref = Evaluator(p)
    for X in np.random.default_rng(2).random((20, 50, 3)):
        ref(X)
    # the members are non-dominated, and the extremes of the full archive survive
    c = p.comparator
    F = ev.archive.F
    assert not c.dominates(F[:, None], 0, F[None], 0).any()
    assert ref.archive.F[:, 0].min() == F[:, 0].min() and ref.archive.F[:, 1].min() == F[:, 1].min()
    assert len(ref.archive) > 10


# runs


def test_random_search_front_end_to_end():
    p = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), two_obj, n_obj=2, name="two")
    r = ypea.RandomSearch(pop_size=20, max_nfe=500).solve(p, seed=1, keep_evaluations=True)
    assert r.best is None and r.best_feasible is None and r.error is None
    ev = r.evaluations
    assert len(ev) == 500 and ev.nfe.tolist() == list(range(1, 501))
    assert rows(r.front.obj_values, r.front.violations) == archive_ref(ev.obj_values, ev.violations, [1, 1])
    assert np.all(np.diff(r.front.obj_values[:, 0]) >= 0)  # sorted by the first objective
    assert np.allclose(r.front[0].solution.x, r.front.solutions[0].x) and r.front.feasible.all()
    h = r.history
    assert h.archive_size[-1] == len(r.front) and np.isnan(h.best_obj_value).all()
    assert h.front_at(500).sum() == len(r.front)
    with pytest.raises(ValueError, match="front_at"):
        h.at_nfe([100])
    d = r.to_dict()
    assert d["best"] is None and len(d["front"]["obj_values"]) == len(r.front) and "removed_nfe" in d["trace"]
    assert "front: " in r.summary()
    # reproducible and the same as the vectorized problem
    q = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), two_obj_vec, n_obj=2, vectorized=True)
    r2 = ypea.RandomSearch(pop_size=20, max_nfe=500).solve(q, seed=1)
    assert np.allclose(r2.front.obj_values, r.front.obj_values)
    r3 = pickle.loads(pickle.dumps(ypea.RandomSearch(pop_size=20, max_nfe=500, max_archive=7))).solve(p, seed=1)
    assert len(r3.front) <= 7


def test_maximization_and_mixed_senses():
    def f(sol):
        return [-sol.x[0], sol.x[0] ** 2]

    p = ypea.Problem(ypea.Real("x", shape=1, lb=-1, ub=1), f, sense=["max", "max"])
    r = ypea.RandomSearch(max_nfe=300).solve(p, seed=0)
    # maximize -x and x^2: the front is x in [-1, 0]
    assert np.all(np.array([s.x for s in r.front.solutions]) <= 0)
    assert np.all(np.diff(r.front.obj_values[:, 0]) <= 0)  # best of the first objective first


def test_single_objective_algorithms_refuse_several_objectives():
    p = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), schaffer, n_obj=2)
    with pytest.raises(ValueError, match="scalarize"):
        ypea.PSO(max_nfe=100).solve(p)
    with pytest.raises(ValueError, match="scalarize"):
        ypea.GA(max_nfe=100).ask_tell(p)
    with pytest.raises(ValueError, match="target"):
        ypea.RandomSearch(max_nfe=100, target=0.0).solve(p)


def test_callbacks_and_display_with_several_objectives(capsys):
    p = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), schaffer, n_obj=2)
    ypea.RandomSearch(max_iter=2, pop_size=5, display=True).solve(p, seed=0, callbacks=[ypea.callbacks.Print()])
    out = capsys.readouterr().out
    assert "Front size" in out and "front size" in out


def test_ask_tell_with_vectors():
    p = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), two_obj, n_obj=2)
    alg = ypea.RandomSearch(pop_size=10, max_nfe=95)
    run = alg.ask_tell(p, seed=4, keep_evaluations=True)
    told = []
    while not run.done:
        vals = [two_obj(s) for s in run.ask()]
        told += vals
        run.tell(vals)
    r, s = run.result(), alg.solve(p, seed=4)
    assert r.front.obj_values.tolist() == s.front.obj_values.tolist()
    assert r.evaluations.obj_values.tolist() == np.array(told).tolist()


def test_scalarize():
    p = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), two_obj, sense=["min", "max"], n_obj=2)
    sol = p.decode_one(np.array([0.25, 0.5, 0.0]))
    f = two_obj(sol)
    ws = ypea.scalarize(p, [0.3, 0.7])
    assert ws.n_obj == 1 and ws.sense == "min"
    assert ws.evaluate(sol)[0] == pytest.approx(0.3 * f[0] - 0.7 * f[1])
    tc = ypea.scalarize(p, [0.3, 0.7], "tchebycheff", ideal=[0, 2])
    assert tc.evaluate(sol)[0] == pytest.approx(max(0.3 * abs(f[0]), 0.7 * abs(f[1] - 2)))
    # vectorized, constraints pass through, pickling, and a single objective run
    q = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), lambda s: (two_obj_vec(s), s.x[:, 0] - 0.5),
                     n_obj=2, vectorized=True)
    qs = pickle.loads(pickle.dumps(ypea.scalarize(ypea.Problem(p.vars, two_obj, n_obj=2), [1, 1])))
    assert qs.evaluate(sol)[0] == pytest.approx(sum(f))
    r = ypea.PSO(max_nfe=500).solve(ypea.scalarize(q, [1, 1]), seed=0)
    assert r.best.feasible and r.best.solution.x[0] <= 0.5
    with pytest.raises(ValueError, match="ideal"):
        ypea.scalarize(p, [1, 1], "tchebycheff")
    with pytest.raises(ValueError, match="weights"):
        ypea.scalarize(p, [1, -1])
    with pytest.raises(ValueError, match="one objective"):
        ypea.scalarize(ypea.Problem(p.vars, lambda s: 0.0), [1])


def test_experiment_report_and_tune_with_several_objectives(tmp_path):
    p = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), two_obj, n_obj=2, name="two")
    with pytest.raises(ValueError, match="pareto_front.*ref_point"):
        ypea.Experiment([ypea.RandomSearch(pop_size=10)], [p], runs=2, max_nfe=100)
    p.info["ref_point"] = [1.1, 2.2]  # no true front: hypervolume only, on the raw values
    exp = ypea.Experiment([ypea.RandomSearch(pop_size=10)], [p], runs=2, max_nfe=100, seed=0, output=tmp_path)
    rep = exp.run().report()
    recs = rep.records("RS", "two")
    assert len(recs) == 2 and recs[0]["result"]["best"] is None and recs[0]["indicators"]["igd"] is None
    front = np.array(recs[0]["result"]["front"]["obj_values"])
    assert rep.values("RS", "two", "hypervolume")[0] == pytest.approx(ypea.indicators.hypervolume(front, [1.1, 2.2]))
    assert rep.table().data["describe"]["two"]["RS"]["mean"] == rep.values("RS", "two", "hypervolume").mean()
    with pytest.raises(ValueError, match="no true front"):
        rep.table(metric="igd")
    with pytest.raises(ValueError, match="single objective"):
        ypea.tune(ypea.RandomSearch, [p], budget=10, max_nfe_per_run=50)


def test_operator_settings_in_stored_runs(tmp_path):
    import json

    from ypea.operators import SBX, NativeCrossover, uniform_crossover

    p = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), lambda s: float(np.sum(s.x**2)), name="sphere")
    ga = ypea.GA(crossover=SBX(eta=15), pop_size=10)
    json.dumps(ypea.GA(crossover=SBX(), max_nfe=100).solve(p).to_dict())
    exp = ypea.Experiment([ga], [p], runs=2, max_nfe=100, seed=0, output=tmp_path)
    exp.run()
    stamp = sorted(f.stat().st_mtime_ns for f in tmp_path.rglob("*.json") if f.name != "metadata.json")
    ypea.Experiment([ypea.GA(crossover=SBX(eta=15), pop_size=10)], [p], runs=2, max_nfe=100, seed=0,
                    output=tmp_path).run()  # resumes: the stored keys match, nothing runs again
    assert stamp == sorted(f.stat().st_mtime_ns for f in tmp_path.rglob("*.json") if f.name != "metadata.json")
    assert repr(NativeCrossover(uniform_crossover, types=ypea.Integer)) == (
        "NativeCrossover(fn=ypea.operators.uniform_crossover, types=(ypea.variables.Integer,))")


def test_front_plot_accepts_a_result():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from ypea.benchmarks.multi import pareto_front
    p = ypea.Problem(ypea.Real("x", shape=2, lb=0, ub=1), lambda s: [s.x[0], 1 - s.x[0] + s.x[1]], n_obj=2)
    r = ypea.RandomSearch(max_nfe=200).solve(p, seed=1)
    ax = ypea.plot.front(r, r.front, true_front=pareto_front("zdt1", 50))
    assert len(ax.collections) == 3
