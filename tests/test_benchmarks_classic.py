import pickle

import numpy as np
import pytest

import ypea
from ypea.benchmarks.classic import REGISTRY, michalewicz, michalewicz_optimum, quartic_noisy
from ypea.benchmarks.problems import classic, names, problem
from ypea.benchmarks.transforms import random_rotation, random_shift

DIMS = (2, 10, 30)
MODES = [(False, False), (True, False), (False, True), (True, True)]


def modes(name):
    return MODES if REGISTRY[name].transformable else MODES[:1]


CASES = [(n, d, s, r) for n in REGISTRY for d in DIMS for s, r in modes(n)]


def close(f, f_opt):
    return abs(f - f_opt) <= 1e-12 * max(1.0, abs(f_opt))


def value(p, x):
    return np.asarray(p.objective(ypea.Solution(x=np.atleast_2d(x))), dtype=float)


@pytest.mark.parametrize("name,dim,shift,rotate", CASES)
def test_optimum_value(name, dim, shift, rotate):
    p = problem(name, dim, shift=shift, rotate=rotate, seed=3)
    x = p.info["x_opt"]
    assert x.shape == (dim,)
    assert close(value(p, x)[0], p.optimum)
    if not (shift or rotate):
        assert close(float(REGISTRY[name].fn(x)), p.optimum)
    assert np.all(x >= p.vars[0].lb) and np.all(x <= p.vars[0].ub)


@pytest.mark.parametrize("name,dim,shift,rotate", [c for c in CASES if c[1] == 10])
def test_no_point_below_optimum(name, dim, shift, rotate):
    p = problem(name, dim, shift=shift, rotate=rotate, seed=5)
    rng = np.random.default_rng(0)
    v = p.vars[0]
    X = v.lb + (v.ub - v.lb) * rng.random((5000, dim))
    X = np.vstack([X, p.info["x_opt"] + 1e-3 * rng.standard_normal((500, dim))])
    assert np.all(value(p, X) >= p.optimum - 1e-9 * max(1.0, abs(p.optimum)))


@pytest.mark.parametrize("name", list(REGISTRY))
def test_vectorized_matches_rows(name):
    b = REGISTRY[name]
    X = b.lb + (b.ub - b.lb) * np.random.default_rng(1).random((4, 3, 7))
    batch = b.fn(X)
    assert batch.shape == (4, 3)
    rows = np.array([[b.fn(x) for x in plane] for plane in X])
    np.testing.assert_allclose(batch, rows, rtol=1e-13, atol=0)


def test_michalewicz_literature_values():
    for d, f in [(2, -1.8013), (5, -4.687658), (10, -9.66015)]:
        assert abs(michalewicz(michalewicz_optimum(d)) - f) < 1e-4
    with pytest.raises(ValueError):
        problem("michalewicz", 5, shift=True)


def test_quartic_noisy():
    x = np.zeros((6, 3))
    f = quartic_noisy(x, np.random.default_rng(0))
    assert f.shape == (6,) and np.all((f >= 0) & (f < 1))


def test_transforms():
    rng = np.random.default_rng(0)
    R = random_rotation(7, rng)
    np.testing.assert_allclose(R @ R.T, np.eye(7), atol=1e-12)
    o = random_shift(-100, 100, rng, fraction=0.8)
    assert -80 <= o <= 80


def test_seeds_and_names():
    a = problem("rastrigin", 10, shift=True, rotate=True, seed=1)
    b = problem("rastrigin", 10, shift=True, seed=1)
    c = problem("rastrigin", 10, rotate=True, seed=1)
    np.testing.assert_array_equal(a.info["x_opt"], b.info["x_opt"])
    np.testing.assert_array_equal(a.info["rotate"], c.info["rotate"])
    assert a.name == "rastrigin_d10_shifted_rotated_s1"
    assert problem("rastrigin", 30).name == "rastrigin_d30"
    assert not np.array_equal(problem("rastrigin", 10, shift=True, seed=2).info["x_opt"], b.info["x_opt"])
    # info["shift"] is o with z = R (x - o) + 0 for a function whose base optimum is 0
    x = np.random.default_rng(2).random(10)
    np.testing.assert_allclose(value(a, x)[0], REGISTRY["rastrigin"].fn(a.info["rotate"] @ (x - a.info["shift"])))


def test_bounds_and_suite():
    p = problem("sphere", 4, bounds=(-1, 2))
    np.testing.assert_array_equal(p.vars[0].lb, -np.ones(4))
    assert len(classic(5)) == len(names()) == len(REGISTRY)
    assert len(classic(5, shift=True)) == len(REGISTRY) - 1
    assert [q.name for q in classic(3, names=["ackley", "levy"])] == ["ackley_d3", "levy_d3"]
    with pytest.raises(ValueError):
        problem("nope", 3)


def test_pickle():
    for p in classic(5, rotate=True, shift=True) + [problem("michalewicz", 4)]:
        q = pickle.loads(pickle.dumps(p))
        x = np.random.default_rng(0).random((3, 5 if "michalewicz" not in p.name else 4))
        np.testing.assert_array_equal(value(p, x), value(q, x))


@pytest.mark.parametrize("name", ["sphere", "rastrigin", "rosenbrock", "happy_cat"])
def test_de_runs(name):
    p = problem(name, 5, shift=True, rotate=True, seed=0)
    r = ypea.DE(max_nfe=2000).solve(p, seed=0)
    assert r.nfe <= 2000
    assert r.best.obj_value >= p.optimum - 1e-9
