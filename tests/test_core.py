import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from problems import neg_sphere_obj, sphere_obj

import ypea
from ypea.comparison import Comparator


def test_comparator_min_max_and_feasibility():
    c = Comparator("min")
    F = np.array([[3.0], [1.0], [2.0], [0.5]])
    V = np.array([0.0, 0.0, 0.0, 1.0])
    assert c.argsort(F, V).tolist() == [1, 2, 0, 3]  # infeasible last despite best f
    assert not c.better(F[[0]], V[[0]], F[[0]], V[[0]])[0]  # ties are not improvements
    assert Comparator("max").argsort(F, V).tolist() == [0, 2, 1, 3]
    p = Comparator("min", "penalty", penalty=10)
    assert p.argsort(F, V).tolist() == [1, 2, 0, 3]
    assert c.ranks(np.array([[1.0], [1.0], [0.0]]), np.zeros(3)).tolist() == [1, 1, 0]


def test_same_seed_same_result_and_reusable(sphere10):
    pso = ypea.PSO(max_iter=20, pop_size=20)
    a = pso.solve(sphere10, seed=5)
    b = pso.solve(sphere10, seed=5)  # same object, run again
    c = ypea.PSO(max_iter=20, pop_size=20).solve(sphere10, seed=6)
    assert a.best.obj_value == b.best.obj_value
    assert np.array_equal(a.history.best_obj_value, b.history.best_obj_value)
    assert a.best.obj_value != c.best.obj_value


def test_pickle_round_trip(sphere10):
    pso = ypea.PSO(max_iter=5, pop_size=10)
    p2, alg2 = pickle.loads(pickle.dumps((sphere10, pso)))
    assert alg2.solve(p2, seed=1).best.obj_value == pso.solve(sphere10, seed=1).best.obj_value
    pickle.dumps(pso.solve(sphere10, seed=1))  # results too


@pytest.mark.parametrize("max_nfe", [1, 37, 250])
def test_max_nfe_is_never_exceeded(sphere10, max_nfe):
    r = ypea.PSO(max_iter=None, pop_size=20, max_nfe=max_nfe).solve(sphere10, seed=0)
    assert r.nfe == max_nfe and r.stop_reason == "max_nfe"
    assert r.history.trace_nfe.max() <= max_nfe


def test_best_and_trace_are_tracked_per_evaluation(sphere10):
    r = ypea.SA(max_iter=10).solve(sphere10, seed=2)
    h = r.history
    assert np.all(np.diff(h.trace_obj_value) < 0) and np.all(np.diff(h.trace_nfe) > 0)
    assert h.trace_obj_value[-1] == r.best.obj_value == h.best_obj_value[-1]
    assert np.all(np.diff(h.best_obj_value) <= 0)
    assert h.at_nfe([h.trace_nfe[-1]])[0] == r.best.obj_value
    assert np.isnan(h.at_nfe([0])[0])


def test_maximization():
    p = ypea.Problem(ypea.Real("x", shape=5, lb=-10, ub=10), neg_sphere_obj, sense="max")
    r = ypea.PSO(max_iter=100, pop_size=30, w=0.5, wdamp=1, c1=1, c2=2).solve(p, seed=1)
    assert -1e-6 < r.best.obj_value <= 0
    m = ypea.Problem(p.vars, sphere_obj, sense="min")
    r2 = ypea.PSO(max_iter=100, pop_size=30, w=0.5, wdamp=1, c1=1, c2=2).solve(m, seed=1)
    assert r2.best.obj_value == -r.best.obj_value  # max of -f mirrors min of f exactly
    assert np.all(np.diff(r.history.best_obj_value) >= 0)


def test_constraints(constrained):
    r = ypea.DE(max_iter=200, pop_size=30).solve(constrained, seed=1)
    assert r.best.feasible
    assert np.allclose(r.best.solution.x, [1.5, 0.5], atol=1e-2)
    pen = ypea.Problem(constrained.vars, constrained.objective, constraint_handling="penalty")
    r2 = ypea.DE(max_iter=200, pop_size=30).solve(pen, seed=1)
    assert abs(r2.best.obj_value - 0.5) < 1e-2


def test_vectorized_and_parallel_match_plain(sphere10):
    vec = ypea.Problem(sphere10.vars, sphere_obj, vectorized=True)
    alg = ypea.DE(max_iter=15, pop_size=20)
    a = alg.solve(sphere10, seed=3).best.obj_value
    b = alg.solve(vec, seed=3).best.obj_value
    with ThreadPoolExecutor(2) as pool:
        c = alg.solve(sphere10, seed=3, executor=pool).best.obj_value
    assert a == pytest.approx(b, rel=1e-12) and a == c


def test_target_and_callback_stop(sphere10):
    r = ypea.PSO(max_iter=500, pop_size=20, target=1.0).solve(sphere10, seed=0)
    assert r.stop_reason == "target" and r.best.obj_value <= 1.0
    r = ypea.PSO(max_iter=500, pop_size=20).solve(sphere10, seed=0, callbacks=[lambda a: a.iter >= 3])
    assert r.stop_reason == "callback" and r.iterations == 3


def test_problem_validation_and_encode(mixed):
    with pytest.raises(ValueError):
        ypea.Problem([ypea.Real("x"), ypea.Real("x")], sphere_obj)
    with pytest.raises(ValueError):  # partitions cannot be vectorized
        ypea.Problem(ypea.Partition("p", n=4, k=2), sphere_obj, vectorized=True)
    sol = mixed.decode_one(np.random.default_rng(0).random(mixed.code_size))
    again = mixed.decode_one(mixed.encode(sol))
    assert mixed.objective(sol) == pytest.approx(mixed.objective(again))


def test_settings_are_validated():
    with pytest.raises(ValueError):
        ypea.PSO(pop_size=0)
    with pytest.raises(ValueError):
        ypea.PSO(max_iter=None)
    with pytest.raises(ValueError):
        ypea.DE("DE/best/9/bin", pop_size=10)
    with pytest.raises(ValueError):
        ypea.DE("DE/nonsense/1/bin")


def test_convergence_plot(sphere10):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    a = ypea.PSO(max_iter=10, pop_size=10).solve(sphere10, seed=0)
    b = ypea.DE(max_iter=10, pop_size=10).solve(sphere10, seed=0)
    ax = ypea.plot.convergence(a, b)
    assert len(ax.lines) == 2 and ax.get_yscale() == "log"
    ypea.plot.convergence(a, x="iter")
