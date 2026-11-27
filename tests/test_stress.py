"""Edge cases every algorithm must survive: unusual sizes, hostile objectives, and the
equivalence of plain, vectorized and parallel evaluation."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_algorithms import NAMES, make

import ypea


def sphere_rows(sol):
    return np.sum(sol.x**2, axis=-1)


def flat(sol):
    return 1.0


def half_infinite(sol):
    return np.inf if sol.x[0] < 0 else float(np.sum(sol.x**2))


def half_nan(sol):
    return np.nan if sol.x[0] < 0 else float(np.sum(sol.x**2))


def never_feasible(sol):
    # minimum violation is 1, at x = 0
    return float(np.sum(sol.x)), 1 + float(np.sum(sol.x**2))


def real(d):
    return ypea.Real("x", shape=d, lb=-1, ub=1)


def check_state(alg, result):
    """Invariants after any run."""
    assert np.all(np.isfinite(alg.best.X)) and np.all((alg.best.X >= 0) & (alg.best.X <= 1))
    h = result.history
    assert len(h.iteration) == result.iterations + 1
    assert np.all(np.diff(h.nfe) >= 0) and h.nfe[-1] == result.nfe
    if len(h.trace_nfe):
        assert np.all(np.diff(h.trace_nfe) > 0) and h.trace_nfe[-1] <= result.nfe
        assert h.trace_obj_value[-1] == result.best.obj_value


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("d", [1, 30])
def test_dimensions(name, d):
    p = ypea.Problem(real(d), sphere_rows)
    alg = make(name, max_nfe=600)
    r = alg.solve(p, seed=1)
    check_state(alg, r)
    assert r.nfe <= 600 and np.isfinite(r.best.obj_value)


@pytest.mark.parametrize("name", NAMES)
def test_flat_objective(name):
    alg = make(name, max_nfe=500)
    r = alg.solve(ypea.Problem(real(4), flat), seed=2)
    check_state(alg, r)
    assert r.best.obj_value == 1.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("objective", [half_infinite, half_nan])
def test_infinite_and_nan_values(name, objective):
    alg = make(name, max_nfe=800)
    r = alg.solve(ypea.Problem(real(3), objective), seed=3)
    check_state(alg, r)
    assert np.isfinite(r.best.obj_value) and r.best.solution.x[0] >= 0


@pytest.mark.parametrize("name", NAMES)
def test_never_feasible(name):
    alg = make(name, max_nfe=1500)
    r = alg.solve(ypea.Problem(real(3), never_feasible), seed=4)
    check_state(alg, r)
    # one constraint, so the raw violation is the scaled one times the scale
    assert not r.best.feasible and 1 <= r.best.violation * r.constraint_scale[0] < 2


@pytest.mark.parametrize("name", NAMES)
def test_plain_vectorized_and_threads_agree(name):
    # 5 components: NumPy sums them in the same order row by row and in batch
    plain = ypea.Problem(real(5), sphere_rows)
    vec = ypea.Problem(real(5), sphere_rows, vectorized=True)
    alg = make(name, max_nfe=700)
    a = alg.solve(plain, seed=5)
    b = alg.solve(vec, seed=5)
    with ThreadPoolExecutor(3) as pool:
        c = alg.solve(plain, seed=5, executor=pool)
    assert a.best.obj_value == b.best.obj_value == c.best.obj_value
    assert np.array_equal(a.history.trace_nfe, b.history.trace_nfe)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("pop_size", [2, 3, 5])
def test_small_populations(name, pop_size):
    try:
        alg = make(name, max_nfe=300, pop_size=pop_size)
    except ValueError:
        return  # this algorithm needs a larger population, and says so
    r = alg.solve(ypea.Problem(real(4), sphere_rows), seed=6)
    check_state(alg, r)
    assert r.nfe == 300 or r.stop_reason in ("no_evaluations", "max_restarts")


@pytest.mark.parametrize("name", NAMES)
def test_max_time_and_target(name):
    p = ypea.Problem(real(4), sphere_rows)
    r = make(name, max_time=0.2).solve(p, seed=7)
    assert r.stop_reason in ("max_time", "no_evaluations", "max_restarts")
    r = make(name, max_nfe=50_000, target=1e-2).solve(p, seed=7)
    assert r.stop_reason == "target" or r.nfe == 50_000


@pytest.mark.parametrize("name", NAMES)
def test_run_state_does_not_leak_between_problems(name):
    alg = make(name, max_nfe=400)
    small = ypea.Problem(real(2), sphere_rows)
    big = ypea.Problem(real(12), sphere_rows)
    first = alg.solve(small, seed=8).best.obj_value
    alg.solve(big, seed=8)
    assert alg.solve(small, seed=8).best.obj_value == first
