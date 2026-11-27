"""The best feasible solution, the evaluation log, and equality constraints."""

import numpy as np
import pytest

import ypea
from ypea.problem import equality_violation


def tricky(sol):
    # rounded values give ties, and the penalty makes infeasible solutions attractive
    x = sol.x
    return float(np.round(np.sum(x**2), 2)), float(np.round(0.5 - x[0], 2))


def first_best_feasible(ev, sign):
    """Plain loop over the evaluations: the first feasible row with the best value."""
    best, trace = None, []
    for k, (f, v) in enumerate(zip(ev.obj_values.tolist(), ev.violations.tolist(), strict=True)):
        if v <= 0 and (best is None or sign * f < sign * ev.obj_values[best]):
            best = k
            trace.append((k + 1, f))
    return best, trace


@pytest.mark.parametrize("constraints", ["feasibility", "penalty"])
@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("name", ["GA", "PSO", "DE"])
def test_best_feasible_matches_brute_force(constraints, sense, name):
    obj = tricky if sense == "min" else (lambda s: (-tricky(s)[0], tricky(s)[1]))
    p = ypea.Problem(ypea.Real("x", shape=3, lb=-1, ub=1), obj, sense=sense, constraints=constraints, penalty=0.5)
    r = getattr(ypea, name)(max_nfe=700, pop_size=30).solve(p, seed=3, keep_evaluations=True)
    ev = r.evaluations
    k, trace = first_best_feasible(ev, 1 if sense == "min" else -1)
    assert r.best_feasible.obj_value == ev.obj_values[k]
    assert r.best_feasible.position.tolist() == ev.positions[k].tolist()
    h = r.history
    assert list(zip(h.trace_feasible_nfe.tolist(), h.trace_feasible_obj_value.tolist(), strict=True)) == trace
    if constraints == "feasibility":
        assert r.best_feasible.obj_value == r.best.obj_value
    d = r.to_dict()
    assert d["best_feasible"]["feasible"] and len(d["evaluations"]["nfe"]) == 700


def test_penalty_best_can_be_infeasible():
    p = ypea.Problem(ypea.Real("x", shape=3, lb=-1, ub=1), tricky, constraints="penalty", penalty=0.01)
    r = ypea.PSO(max_nfe=1000).solve(p, seed=0)
    assert not r.best.feasible and r.best_feasible.feasible
    assert r.best_feasible.obj_value > r.best.obj_value
    assert "best feasible objective value" in r.summary()
    q = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), lambda s: (s.x[0], 1.0))
    r = ypea.RandomSearch(max_nfe=50).solve(q, seed=0)
    assert r.best_feasible is None and "none found" in r.summary() and r.to_dict()["best_feasible"] is None


def test_log_matches_ask_and_tell():
    p = ypea.Problem(ypea.Real("x", shape=3, lb=-1, ub=1), None)
    run = ypea.DE(max_nfe=333, pop_size=20).ask_tell(p, seed=1, keep_evaluations=True)
    codes, told = [], []
    while not run.done:
        sols = run.ask()
        codes.append(run.codes)
        vals = [tricky(s) for s in sols]
        told += vals
        run.tell(vals)
    r = run.result()
    ev = r.evaluations
    assert len(ev) == 333 and ev.nfe.tolist() == list(range(1, 334))
    assert ev.positions.tolist() == np.concatenate(codes).tolist()
    assert ev.obj_values.tolist() == [f for f, _ in told]
    assert ev.violations.tolist() == [max(v, 0.0) / r.constraint_scale[0] for _, v in told]
    feas = ev[ev.feasible]
    assert feas.violations.max() <= 0 and len(feas) == int(ev.feasible.sum())
    srt = ev[np.argsort(ev.obj_values, kind="stable")]
    assert np.all(np.diff(srt.obj_values) >= 0) and len(ev[5]) == 1


def test_log_is_off_by_default_and_skips_budget_rows():
    p = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), lambda s: float(s.x[0]))
    assert ypea.GA(max_nfe=50).solve(p, seed=0).evaluations is None
    r = ypea.GA(max_nfe=55, pop_size=20).solve(p, seed=0, keep_evaluations=True)
    assert len(r.evaluations) == 55 and np.all(np.isfinite(r.evaluations.obj_values))


# equality constraints


def test_equality_violation_rule():
    assert equality_violation(0.0) == 0.0
    assert equality_violation(1e-4) == 0.0  # within atol
    assert equality_violation(-0.5) == pytest.approx(0.5 - 1e-4 - 1e-6)
    assert equality_violation([0.1, -0.2], (0.0, 0.1, np.array([1.0, 3.0]))) == pytest.approx(0.0 + 0.0 + 0.0)
    assert equality_violation([0.2, -0.4], (0.0, 0.1, np.array([1.0, 3.0]))) == pytest.approx(0.1 + 0.1)
    v = equality_violation(np.array([[0.0, 1.0], [2.0, 0.0]]), (0.5, 0.0, 1.0), n=2)
    assert v.tolist() == [0.5, 1.5]
    with pytest.raises(ValueError, match="equality_scale"):
        equality_violation([1.0, 2.0, 3.0], (0.0, 0.1, np.array([1.0, 3.0])))


def circle(sol):
    x, y = sol.x
    return x + y, None, x**2 + y**2 - 1  # minimize x + y on the unit circle


def test_equality_single_objective_run():
    p = ypea.Problem(ypea.Real("x", shape=2, lb=-2, ub=2), circle)
    assert p.evaluate(ypea.Solution(x=np.array([1.0, 0.0])))[1] == 0.0
    assert p.evaluate(ypea.Solution(x=np.array([2.0, 0.0])))[1] == pytest.approx(3 - 1e-4 - 1e-6)
    r = ypea.DE(max_nfe=20000).solve(p, seed=0)
    assert r.best.feasible and -1.4145 < r.best.obj_value < -1.3
    q = ypea.Problem(ypea.Real("x", shape=2, lb=-2, ub=2),
                     lambda s: (s.x.sum(axis=1), np.zeros((len(s.x), 0)), (s.x**2).sum(axis=1) - 1), vectorized=True)
    rq = ypea.DE(max_nfe=20000).solve(q, seed=0)
    assert rq.best.obj_value == pytest.approx(r.best.obj_value) and rq.best.violation == r.best.violation


def test_equality_with_inequalities_scale_and_tolerances():
    def f(sol):
        x = sol.x
        return float(x[0]), [x[1] - 1], [x[0] - x[2], 1000 * (x[1] - x[2])]

    p = ypea.Problem(ypea.Real("x", shape=3, lb=-2, ub=2), f, equality_atol=0.0, equality_scale=[1, 1000])
    sol = ypea.Solution(x=np.array([0.5, 1.5, 0.5]))
    # inequality 0.5, equality 1: |0| -> 0, equality 2: 1000 - 1e-6 * 1000
    assert p.evaluate(sol)[1] == pytest.approx(0.5 + 1000 - 1e-3)
    with pytest.raises(ValueError, match="equality"):
        ypea.Problem(ypea.Real("x"), f, equality_atol=-1)


def test_equality_multi_objective_and_ask_tell():
    def f(sol):
        x = sol.x
        return [x[0], 1 - x[0]], [], x[0] - x[1]

    p = ypea.Problem(ypea.Real("x", shape=2, lb=0, ub=1), f, n_obj=2, equality_atol=0.05)
    r = ypea.RandomSearch(max_nfe=400).solve(p, seed=0)
    assert r.front.feasible.all()
    assert np.all(np.abs([s.x[0] - s.x[1] for s in r.front.solutions]) <= 0.05)
    run = ypea.RandomSearch(max_nfe=400).ask_tell(ypea.Problem(p.vars, None, n_obj=2, equality_atol=0.05), seed=0)
    while not run.done:
        run.tell([f(s) for s in run.ask()])
    assert run.result().front.obj_values.tolist() == r.front.obj_values.tolist()
