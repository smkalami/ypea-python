"""Checks that target one algorithm's own behavior."""

import numpy as np
import pytest

import ypea
from ypea.benchmarks import rosenbrock


def rosen(sol):
    return rosenbrock(sol.x)


def test_cmaes_learns_rosenbrock():
    # only an adapted covariance matrix solves Rosenbrock quickly
    p = ypea.Problem(ypea.Real("x", shape=10, lb=-5, ub=10), rosen)
    r = ypea.CMAES(max_nfe=20_000, pop_size=40).solve(p, seed=0)
    assert r.best.obj_value < 1e-10
    assert ypea.CMAES(max_iter=1).solve(p, seed=0).nfe == 1 + 4 + int(3 * np.log(10))


def test_iwo_population_grows_to_max_and_step_shrinks(sphere10):
    alg = ypea.IWO(max_iter=30, pop_size=20, initial_pop_size=5)
    alg.solve(sphere10, seed=0)
    assert len(alg.pop) == 20
    assert np.isclose(alg.step(), alg.final_step_size)  # the last iteration uses the final step


def test_iwo_auto_fitness_switches_to_ranks_with_constraints(constrained, sphere10):
    a = ypea.IWO(max_iter=2)
    a.solve(sphere10, seed=0)
    assert not a.constrained
    a.solve(constrained, seed=0)
    assert a.constrained


def plateau(sol):
    return float(np.sum(sol.x**2)), 1.0  # every solution has the same violation


def test_sa_follows_the_comparator_on_a_violation_plateau():
    # equal violations: the objective decides, so a cold SA must not drift to worse objectives
    p = ypea.Problem(ypea.Real("x", shape=3, lb=-5, ub=5), plateau)
    alg = ypea.SA(pop_size=1, move_count=1, max_sub_iter=1, max_iter=2000, initial_temp=1e-9, final_temp=1e-12)
    alg.solve(p, seed=0)
    assert alg.pop.F[0, 0] < 1e-3  # about 32 before the fix


def test_sa_temperature_stops_at_final_temp(sphere10):
    alg = ypea.SA(max_time=0.5, final_temp=1.0)
    alg.solve(sphere10, seed=0)
    assert alg._temp >= 1.0


def test_iwo_step_schedule_ends_with_the_run(sphere10):
    # the schedule follows progress(), so it reaches the final step at the end of the run
    alg = ypea.IWO(max_nfe=20_000)
    steps = []
    alg.solve(sphere10, seed=0, callbacks=[lambda a: steps.append(a.step())])
    final = alg.final_step_size
    assert steps[len(steps) // 2] > 2 * final  # not at the final step halfway
    assert abs(steps[-1] - final) < 0.05 * (alg.step_size - final)


def test_iwo_infinite_weeds_get_no_extra_seeds():
    alg = ypea.IWO(max_iter=1)
    alg.solve(ypea.Problem(ypea.Real("x"), lambda s: float(s.x)), seed=0)
    alg.pop.F[:] = np.array([[1.0], [2.0], [3.0], [np.inf]] + [[2.0]] * (len(alg.pop) - 4))
    r = alg.seed_ratio()
    assert r[0] == 1 and r[2] == 0 and r[3] == 0


def test_ca_with_a_small_population_keeps_searching():
    p = ypea.Problem(ypea.Real("x", shape=5, lb=-5, ub=5), lambda s: float(np.sum(s.x**2)))
    r = ypea.CA(pop_size=5, max_nfe=2000).solve(p, seed=0)
    assert r.history.trace_nfe[-1] > 100 and r.best.obj_value < 1.0  # stalled at 28 after 4 evaluations before


def test_plot_steps_hold_after_each_improvement(sphere10):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    ax = ypea.plot.convergence(ypea.HS(max_iter=5).solve(sphere10, seed=0))
    assert ax.lines[0].get_drawstyle() == "steps-post"


def test_input_checks():
    with pytest.raises(ValueError):
        ypea.FixedSum("t", shape=(2, 2), axis=2)
    with pytest.raises(ValueError):
        ypea.Problem(ypea.Real("x"), lambda s: 0.0, constraints="penalty", penalty=-1)
    one = ypea.Problem(ypea.Real("x"), lambda s: np.array([2.0]))
    assert one.evaluate(one.decode_one([0.5])) == (2.0, 0.0)  # a one element array is fine
    def bad_g(sol):
        f = np.sum(sol.x**2, axis=1)
        return f, np.stack([f - 1, -np.ones_like(f)])  # (m, n) instead of (n, m)
    vec = ypea.Problem(ypea.Real("x", shape=2), bad_g, vectorized=True)
    with pytest.raises(ValueError):
        ypea.DE(max_iter=1, pop_size=5).solve(vec, seed=0)


def inf_plateau(sol):
    return np.inf


def test_ipop_fixes():
    from ypea.algorithms.ipop_cmaes import IPOPCMAES

    a = IPOPCMAES(max_nfe=1000, incpopsize=1.1)
    assert "mu" not in a.params() and type(a)(**a.params()).params() == a.params()
    p = ypea.Problem(ypea.Real("x", shape=3), inf_plateau)
    a.solve(p, seed=0)
    assert a.restarts > 0  # an inf plateau triggers restarts
    sizes = [lam for _, lam, _ in a.restart_log]
    assert all(b > c for b, c in zip(sizes[1:], sizes[:-1], strict=False))  # grows every restart
    with pytest.raises(ValueError):
        ypea.CMAES(max_iter=1, x0=[0.5, 0.5]).solve(p, seed=0)


def test_small_bee_populations_still_search(sphere10):
    for t in ("standard", "probabilistic"):
        r = ypea.BA(pop_size=3, type=t, max_iter=50).solve(sphere10, seed=0)
        assert r.stop_reason == "max_iter"
    a = ypea.ABC(pop_size=20, onlooker_count=0, max_iter=1)
    a.solve(sphere10, seed=0)
    assert a._limit == 20 * 10


def test_cmaes_does_not_depend_on_eigenvector_signs(monkeypatch, sphere10):
    # the symmetric root B D B^T z is the same for any sign of each eigenvector
    runs = [ypea.CMAES(max_iter=30).solve(sphere10, seed=3)]
    eigh = np.linalg.eigh
    monkeypatch.setattr(np.linalg, "eigh", lambda C: (lambda w, B: (w, -B))(*eigh(C)))
    runs.append(ypea.CMAES(max_iter=30).solve(sphere10, seed=3))
    assert np.array_equal(runs[0].history.best_obj_value, runs[1].history.best_obj_value)


def test_pso_linear_inertia_and_velocity_limit(sphere10):
    seen = []

    def watch(alg):
        seen.append((alg._w, float(np.abs(alg.velocity).max())))

    ypea.PSO(w=0.9, w_final=0.4, max_velocity=0.1, max_iter=11).solve(sphere10, seed=0, callbacks=[watch])
    assert np.isclose(seen[0][0], 0.9) and np.isclose(seen[-1][0], 0.4)
    assert max(v for _, v in seen) <= 0.1
    # the defaults are unchanged
    a = ypea.PSO(max_nfe=2000).solve(sphere10, seed=1)
    b = ypea.PSO(max_nfe=2000, w_final=None, max_velocity=None).solve(sphere10, seed=1)
    assert np.array_equal(a.history.trace_obj_value, b.history.trace_obj_value)
