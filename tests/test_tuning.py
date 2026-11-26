import json

import numpy as np
import pytest
from problems import neg_sphere_obj

import ypea
from ypea.algorithms import __all__ as ALGORITHMS
from ypea.benchmarks import problem
from ypea.tuning import SPACES, Space, categorical, integer, real, space, sweep, tune
from ypea.tuning.spaces import COMMON

pytest.importorskip("scipy")

CLASSES = [getattr(ypea, n) for n in ALGORITHMS]
PLAIN = (bool, int, float, str)


def test_every_algorithm_has_a_space():
    assert set(SPACES) == set(CLASSES)
    with pytest.raises(KeyError, match="no built in space"):
        space(type("Mine", (ypea.DE,), {}))


@pytest.mark.parametrize("cls", CLASSES, ids=ALGORITHMS)
def test_space_covers_every_constructor_parameter(cls):
    sp = space(cls)
    names = set(cls().params())
    covered = set(sp.params) | set(sp.fixed)
    assert set(sp.params).isdisjoint(sp.fixed)
    assert names - set(COMMON) <= covered, f"not in the space: {names - set(COMMON) - covered}"
    assert "pop_size" in covered and covered <= names
    assert all(isinstance(r, str) and r for r in sp.fixed.values())
    for name, p in sp.params.items():
        if p.kind != "cat":
            lo, hi = p.valid
            assert (lo is None or p.low >= lo) and (hi is None or p.high <= hi), name


@pytest.mark.parametrize("cls", CLASSES, ids=ALGORITHMS)
def test_sampled_settings_are_valid_and_run(cls):
    sp, rng = space(cls), np.random.default_rng(1)
    probs = {k: [1 / len(p.values)] * len(p.values) for k, p in sp.params.items() if p.kind == "cat"}
    settings = [sp.sample(rng) for _ in range(100)]
    settings += [sp.neighbor(rng, s, 0.5, probs) for s in settings[:50]]
    for s in settings:
        assert all(type(v) in PLAIN for v in s.values()), s  # NumPy scalars would fail validation or JSON
        for k, v in s.items():
            p = sp.params[k]
            assert v in p.values if p.kind == "cat" else p.low <= v <= p.high
        cls(**s)  # raises if validate() rejects it
    prob = problem("sphere", 3)
    if cls.objectives == "multi":
        prob = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), lambda sol: sol.x[:, :2], vectorized=True, n_obj=2)
    for s in settings[:2]:
        assert cls(**s, max_nfe=200).solve(prob, seed=0).nfe <= 200



@pytest.mark.parametrize("cls", CLASSES, ids=ALGORITHMS)
def test_defaults_lie_in_the_typical_ranges(cls):
    defaults = cls().params()
    for name, p in space(cls).params.items():
        v = defaults[name]
        assert v is None or (v in p.values if p.kind == "cat" else p.low <= v <= p.high), (name, v)


def test_param_round_trip():
    for p in (real(1e-4, 1, log=True), integer(1, 100, log=True), integer(0, 3), real(-1, 1)):
        for u in np.linspace(0, 1, 11):
            v = p.from_unit(u)
            assert p.low <= v <= p.high
            assert p.from_unit(p.to_unit(v)) == v
    with pytest.raises(ValueError):
        real(0, 1, log=True)


# a toy where one side clearly wins: DE on separable Rastrigin likes a small crossover_prob
TOY = [problem("rastrigin", 5, shift=True, seed=1), problem("rastrigin", 5, shift=True, seed=2)]
TOY_ARGS = {"budget": 150, "space": Space({"crossover_prob": real(0, 1)}), "max_nfe_per_run": 1000,
            "fixed": {"pop_size": 20, "beta_min": 0.2, "beta_max": 0.9}, "seed": 3}


@pytest.fixture(scope="module")
def toy_result():
    return tune(ypea.DE, TOY, **TOY_ARGS)


def test_tune_keeps_the_good_region(toy_result):
    r = toy_result
    first = r.history[0]
    assert first["survivors"] < first["settings"]  # the race eliminated settings
    assert r.best["crossover_prob"] < 0.3
    assert all(e["crossover_prob"] < 0.35 for e in r.elites)
    assert r.runs <= TOY_ARGS["budget"] and r.runs == sum(h["runs"] for h in r.history)
    alg = r.algorithm(max_nfe=500)
    assert isinstance(alg, ypea.DE) and alg.pop_size == 20 and alg.crossover_prob == r.best["crossover_prob"]


def test_tune_is_reproducible(toy_result):
    assert tune(ypea.DE, TOY, **TOY_ARGS).to_dict() == toy_result.to_dict()


def test_tune_does_not_depend_on_n_jobs(toy_result):
    assert tune(ypea.DE, TOY, **TOY_ARGS, n_jobs=2).to_dict() == toy_result.to_dict()


def test_tune_resumes_from_output(tmp_path, toy_result):
    r = tune(ypea.DE, TOY, **TOY_ARGS, output=tmp_path)
    assert r.to_dict() == toy_result.to_dict()
    files = sorted((tmp_path / "runs").glob("*.json"))
    assert len(files) == r.runs
    assert json.loads((tmp_path / "tune.json").read_text("utf-8"))["best"] == r.best
    removed = files[::7]
    for f in removed:
        f.unlink()
    stamps = {f: f.stat().st_mtime_ns for f in files if f not in removed}
    again = tune(ypea.DE, TOY, **TOY_ARGS, output=tmp_path)
    assert again.to_dict() == r.to_dict()
    assert all(f.exists() for f in removed)  # only the missing runs were made again
    assert {f: f.stat().st_mtime_ns for f in stamps} == stamps


def test_tune_with_categories_and_maximization():
    probs = [ypea.Problem(ypea.Real("x", shape=3, lb=-5, ub=5), neg_sphere_obj, "max", name=f"neg{i}")
             for i in range(2)]
    sp = Space({"base_vector": categorical("rand", "best"), "pop_size": integer(10, 40)})
    r = tune(ypea.DE, probs, budget=80, space=sp, max_nfe_per_run=300, first_test=3, seed=0)
    assert r.best["base_vector"] in ("rand", "best") and 10 <= r.best["pop_size"] <= 40
    assert r.runs <= 80


def test_tune_checks_its_arguments():
    sp = Space({"crossover_prob": real(0, 1)})
    with pytest.raises(ValueError, match="too small"):
        tune(ypea.DE, TOY, budget=10, space=sp)
    with pytest.raises(ValueError, match="nothing to tune"):
        tune(ypea.DE, TOY, budget=100, space=sp, fixed={"crossover_prob": 0.1})
    with pytest.raises(ValueError, match="unique"):
        tune(ypea.DE, [TOY[0], TOY[0]], budget=100, space=sp)
    with pytest.warns(UserWarning, match="runs unused"):  # 10 settings keep 5 elites, too many for this budget
        tune(ypea.GA, TOY, budget=200, max_nfe_per_run=200)
    with pytest.raises(ValueError, match="all invalid"):
        tune(ypea.DE, TOY, budget=100, space=Space({"beta_min": real(0.6, 0.9)}), fixed={"beta_max": 0.5})


def test_sweep_grid_list_and_random(tmp_path):
    args = {"runs": 2, "max_nfe": 200, "fixed": {"pop_size": 10}}
    rep = sweep(ypea.DE, {"crossover_prob": [0.1, 0.9]}, TOY[:1], **args)
    assert rep.labels == ["DE(crossover_prob=0.1)", "DE(crossover_prob=0.9)"]
    assert len(rep.values(rep.labels[0], TOY[0].name)) == 2
    rep = sweep(ypea.DE, [{"crossover_prob": 0.2}, {"crossover_prob": 0.4}], TOY[:1], **args)
    assert len(rep.labels) == 2
    rep = sweep(ypea.DE, None, TOY[:1], n=3, output=tmp_path, **args)
    assert rep.labels == ["DE 1", "DE 2", "DE 3"]
    assert all(rep.metadata["algorithms"][k]["params"]["pop_size"] == 10 for k in rep.labels)
    with pytest.raises(ValueError, match="need n"):
        sweep(ypea.DE, None, TOY[:1], **args)

