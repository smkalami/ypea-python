"""The fast paths of the Evaluator and of Problem.decode match plain reference loops exactly."""

import numpy as np
import pytest

import ypea
from ypea.evaluation import Evaluator


def reference_trace(problem, batches, max_nfe=None):
    """Evaluate one solution at a time and compare each with the best so far."""
    cmp = problem.comparator
    best, nfe, trace = None, 0, []
    for X in batches:
        for x in np.clip(X, 0, 1):
            if max_nfe is not None and nfe >= max_nfe:
                break
            f, v = problem.evaluate(problem.decode_one(x))
            f = f if np.isfinite(f) else cmp.worst_value
            v = np.inf if np.isnan(v) else v
            nfe += 1
            if best is None or cmp.better(np.array([[f]]), np.array([v]), best[0], best[1])[0]:
                best = (np.array([[f]]), np.array([v]))
                trace.append((nfe, f, v))
    return trace, best


def tricky(sol):
    # rounded values give many exact ties, and some rows are NaN or infinite
    x = sol.x
    f = float(np.round(np.sum(x**2), 1))
    if x[0] > 0.9:
        f = float("nan")
    elif x[0] < -0.9:
        f = -np.inf
    g = float("nan") if x[1] < -0.9 else float(np.round(x[0] + x[1] + 1.5, 1))
    return f, g


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("constraints", ["feasibility", "penalty"])
@pytest.mark.parametrize("max_nfe", [None, 333])
def test_evaluator_trace_matches_sequential_loop(sense, constraints, max_nfe):
    problem = ypea.Problem(ypea.Real("x", shape=2, lb=-1, ub=1), tricky, sense=sense,
                           constraints=constraints, penalty=3.0, constraint_scaling=False)
    rng = np.random.default_rng(0)
    batches = [rng.uniform(-0.1, 1.1, (n, 2)) for n in (1, 7, 50, 3, 200, 50, 100)]
    ev = Evaluator(problem, max_nfe=max_nfe)
    for X in batches:
        ev(X)
    trace, best = reference_trace(problem, batches, max_nfe)
    assert list(zip(ev.trace_nfe, ev.trace_f, ev.trace_v, strict=True)) == trace
    assert ev.best.F.tolist() == best[0].tolist() and ev.best.V.tolist() == best[1].tolist()


def test_decode_matches_row_indexing(mixed):
    X = np.random.default_rng(1).random((9, mixed.code_size))
    X[0], X[1] = 0.0, 1.0
    sols = mixed.decode(X)
    for x, sol in zip(X, sols, strict=True):
        one = mixed.decode_one(x)
        assert vars(sol).keys() == vars(one).keys()
        for k, v in vars(sol).items():
            w = getattr(one, k)
            if isinstance(v, list):
                assert all(np.array_equal(a, b) for a, b in zip(v, w, strict=True))
            else:
                assert np.array_equal(v, w) and np.asarray(v).dtype == np.asarray(w).dtype
