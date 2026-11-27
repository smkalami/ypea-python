"""Scaling of the constraint violations (Problem(constraint_scaling=...))."""

import numpy as np
import pytest

import ypea
from ypea.algorithms import CMAES, DE, SA


def two_constraints(s):
    # constraints of very different sizes: x0 >= 1 (times 100) and x1 <= 2 (times 0.01)
    return float((s.x ** 2).sum()), [100 * (1 - s.x[0]), 0.01 * (s.x[1] - 2)]


def two_constraints_vec(s):
    x = s.x
    return (x ** 2).sum(axis=1), np.column_stack([100 * (1 - x[:, 0]), 0.01 * (x[:, 1] - 2)])


def problem(obj=two_constraints, **kw):
    return ypea.Problem(ypea.Real("x", 3, -5, 5), obj, **kw)


def raw_violation(p, X):
    return np.array([p.evaluate(s)[1] for s in p.decode(X)])


def test_scaled_violation_is_zero_exactly_when_raw_is_zero():
    p = problem()
    r = DE(max_nfe=600).solve(p, seed=3, keep_evaluations=True)
    ev = r.evaluations
    raw = raw_violation(p, ev.positions)
    assert np.array_equal(ev.violations == 0, raw == 0)
    # each violation is the sum of the parts divided by the scales
    parts = np.array([[max(0.0, 100 * (1 - s.x[0])), max(0.0, 0.01 * (s.x[1] - 2))] for s in p.decode(ev.positions)])
    assert np.allclose(ev.violations, (parts / r.constraint_scale).sum(axis=1))


def test_scale_is_the_largest_first_violation():
    p = problem()
    r = DE(max_nfe=300, pop_size=50).solve(p, seed=1, keep_evaluations=True)
    first = r.evaluations.positions[:50]
    parts = np.array([[max(0.0, 100 * (1 - s.x[0])), max(0.0, 0.01 * (s.x[1] - 2))] for s in p.decode(first)])
    assert np.array_equal(r.constraint_scale, parts.max(axis=0))
    assert r.to_dict()["constraint_scale"] == list(r.constraint_scale)


def test_one_constraint_gives_the_same_search_under_feasibility_rules():
    def one(s):
        return float((s.x ** 2).sum()), 1 - s.x[0]
    on = DE(max_nfe=1000).solve(problem(one), seed=2)
    off = DE(max_nfe=1000).solve(problem(one, constraint_scaling=False), seed=2)
    assert np.array_equal(on.best.position, off.best.position)
    assert on.best.obj_value == off.best.obj_value


def test_off_adds_the_raw_violations():
    p = problem(constraint_scaling=False)
    r = DE(max_nfe=300).solve(p, seed=1, keep_evaluations=True)
    assert r.constraint_scale is None
    assert np.allclose(r.evaluations.violations, raw_violation(p, r.evaluations.positions))


def test_small_first_batch_gets_a_sample_of_20():
    p = problem()
    r = CMAES(max_nfe=500).solve(p, seed=1, keep_evaluations=True)
    assert r.nfe == 500
    first20 = r.evaluations.positions[:20]
    parts = np.array([[max(0.0, 100 * (1 - s.x[0])), max(0.0, 0.01 * (s.x[1] - 2))] for s in p.decode(first20)])
    assert np.array_equal(r.constraint_scale, np.where(parts.max(axis=0) > 0, parts.max(axis=0), 1.0))


def test_sample_respects_the_budget():
    r = SA(max_nfe=12).solve(problem(), seed=1)
    assert r.nfe == 12


def test_unconstrained_problems_draw_no_sample():
    p = ypea.Problem(ypea.Real("x", 3, -5, 5), lambda s: float((s.x ** 2).sum()))
    on = CMAES(max_nfe=300).solve(p, seed=4)
    off = CMAES(max_nfe=300).solve(ypea.Problem(p.vars, p.objective, constraint_scaling=False), seed=4)
    assert on.best.obj_value == off.best.obj_value and on.constraint_scale is None


def test_vectorized_matches_plain():
    a = SA(max_nfe=400).solve(problem(), seed=5)
    b = SA(max_nfe=400).solve(problem(two_constraints_vec, vectorized=True), seed=5)
    assert np.array_equal(a.constraint_scale, b.constraint_scale)
    assert a.best.obj_value == b.best.obj_value


def test_ask_tell_matches_solve():
    p = problem()
    want = CMAES(max_nfe=300).solve(p, seed=7)
    run = CMAES(max_nfe=300).ask_tell(p, seed=7)
    while not run.done:
        run.tell([two_constraints(s) for s in run.ask()])
    got = run.result()
    assert np.array_equal(got.constraint_scale, want.constraint_scale)
    assert got.best.obj_value == want.best.obj_value and got.best.violation == want.best.violation


def test_given_scales_are_used():
    p = problem(constraint_scaling=[100, 0.01])
    r = DE(max_nfe=300).solve(p, seed=1, keep_evaluations=True)
    assert np.array_equal(r.constraint_scale, [100, 0.01])
    parts = np.array([[max(0.0, 1 - s.x[0]), max(0.0, s.x[1] - 2)] for s in p.decode(r.evaluations.positions)])
    assert np.allclose(r.evaluations.violations, parts.sum(axis=1))


def test_given_scales_must_match_the_constraints():
    with pytest.raises(ValueError, match="constraint_scaling has 3 entries"):
        DE(max_nfe=100).solve(problem(constraint_scaling=[1, 2, 3]), seed=1)


@pytest.mark.parametrize("bad", [[], [1, 0], [1, -2], [[1, 2]], [1, np.inf]])
def test_bad_scales_raise(bad):
    with pytest.raises(ValueError, match="constraint_scaling"):
        problem(constraint_scaling=bad)


def test_different_constraint_counts_raise():
    def uneven(s):
        return float(s.x[0]), [s.x[0]] if s.x[1] > 0 else [s.x[0], s.x[1]]
    with pytest.raises(ValueError, match="same number of constraints"):
        DE(max_nfe=100).solve(problem(uneven), seed=1)


def test_non_finite_violations_do_not_spoil_the_scale():
    def wild(s):
        return float((s.x ** 2).sum()), [np.nan if s.x[2] > 4 else 1 - s.x[0], s.x[1] - 2]
    r = DE(max_nfe=300).solve(problem(wild), seed=1)
    assert np.all(np.isfinite(r.constraint_scale)) and np.all(r.constraint_scale > 0)


def test_equalities_are_scaled_after_the_inequalities():
    def eq(s):
        return float((s.x ** 2).sum()), [1 - s.x[0]], [1000 * (s.x[1] - 1)]
    r = DE(max_nfe=300).solve(problem(eq), seed=1)
    assert len(r.constraint_scale) == 2


@pytest.mark.parametrize("setting", [False, [2.0]])
def test_scalarize_keeps_the_setting(setting):
    p = ypea.Problem(ypea.Real("x", 2, 0, 1), lambda s: ([s.x[0], 1 - s.x[0]], s.x[1] - 0.5), n_obj=2,
                     constraint_scaling=setting)
    q = ypea.scalarize(p, [0.5, 0.5])
    assert q.constraint_scaling is setting if setting is False else list(q.constraint_scaling) == setting


@pytest.mark.parametrize("vectorized", [False, True])
def test_race_reports_raw_violation(vectorized):
    from ypea.tuning.race import run_one

    def never(s):  # violation 5 + 10 * x0^2 > 0, so never feasible
        x = s.x
        f, g = (x ** 2).sum(axis=-1), 5 + 10 * x[..., 0] ** 2
        return (f, g) if vectorized else (float(f), float(g))
    p = problem(never, vectorized=vectorized)
    f, v = run_one((DE, {"max_nfe": 300}, p, 1, None, None))
    r = DE(max_nfe=300).solve(p, seed=1)
    assert v == pytest.approx(5 + 10 * r.best.solution.x[0] ** 2) and v >= 5
