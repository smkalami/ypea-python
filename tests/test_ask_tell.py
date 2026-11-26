"""Ask and tell: the same results as solve(), the budget, errors, stopping, and no stray threads."""

import threading

import numpy as np
import pytest
from problems import constrained_obj, sphere_obj
from test_algorithms import NAMES, make

import ypea


def drive(run, objective, vectorized=False):
    """Tell the true objective values until the run ends. Returns the batch sizes asked for."""
    sizes = []
    while not run.done:
        sols = run.ask()
        sizes.append(len(run.codes))
        run.tell(objective(sols) if vectorized else [objective(s) for s in sols])
    return sizes


def same(a, b):
    assert a.nfe == b.nfe and a.iterations == b.iterations and a.stop_reason == b.stop_reason
    assert a.best.obj_value == b.best.obj_value and a.best.violation == b.best.violation
    assert a.best.position.tolist() == b.best.position.tolist()
    for name in ("iteration", "nfe", "best_obj_value", "best_violation", "trace_nfe", "trace_obj_value",
                 "trace_violation"):
        assert getattr(a.history, name).tolist() == getattr(b.history, name).tolist(), name
    assert a.extra.keys() == b.extra.keys()


def no_worker():
    return not any(t.name == "ypea-ask-tell" and t.is_alive() for t in threading.enumerate())


@pytest.mark.parametrize("seed", [0, ypea.StudioRng(7)], ids=["numpy", "studio"])
@pytest.mark.parametrize("name", NAMES)
def test_same_result_as_solve(name, seed, sphere10):
    alg = make(name, max_nfe=300)
    run = alg.ask_tell(sphere10, seed=seed)
    sizes = drive(run, sphere_obj)
    same(run.result(), alg.solve(sphere10, seed=seed))
    assert sum(sizes) == run.result().nfe == 300  # never asked past the budget
    assert no_worker()


@pytest.mark.parametrize("name", ["PSO", "DE/rand/1/bin", "ABC", "IWO", "GA-native", "SA"])
def test_constrained_vectorized_and_mixed(name, constrained, mixed):
    alg = make(name, max_nfe=123)
    same(drive_result(alg, constrained, constrained_obj), alg.solve(constrained, seed=3))
    vec = ypea.Problem(constrained.vars, constrained_obj, vectorized=True)
    same(drive_result(alg, vec, constrained_obj, vectorized=True), alg.solve(vec, seed=3))
    same(drive_result(alg, mixed, mixed.objective), alg.solve(mixed, seed=3))


def drive_result(alg, problem, objective, vectorized=False):
    run = alg.ask_tell(ypea.Problem(problem.vars, None, vectorized=vectorized), seed=3)
    drive(run, objective, vectorized)
    return run.result()


def test_target_callbacks_and_request_stop(sphere10):
    alg = ypea.DE(max_nfe=5000, target=1.0)
    same(drive_result(alg, sphere10, sphere_obj), alg.solve(sphere10, seed=3))

    class Hooks:
        def __init__(self):
            self.calls = []

        def on_start(self, alg):
            self.calls.append("start")

        def __call__(self, alg):
            self.calls.append(alg.iter)
            if alg.iter == 4:
                alg.request_stop("enough")
            return alg.iter == 10

        def on_end(self, alg, result):
            self.calls.append(result.stop_reason)

    a, b = Hooks(), Hooks()
    alg = ypea.PSO(max_iter=50, pop_size=10)
    run = alg.ask_tell(sphere10, seed=1, callbacks=[a])
    drive(run, sphere_obj)
    same(run.result(), alg.solve(sphere10, seed=1, callbacks=[b]))
    assert a.calls == b.calls == ["start", 1, 2, 3, 4, "enough"]


def test_no_objective(sphere10):
    p = ypea.Problem(sphere10.vars, None)
    with pytest.raises(ValueError, match="ask_tell"):
        ypea.PSO(max_iter=2).solve(p)
    with pytest.raises(ValueError, match="no objective"):
        p.evaluate(p.decode_one(np.zeros(10)))


def test_wrong_use(sphere10):
    run = ypea.PSO(max_nfe=40, pop_size=20).ask_tell(sphere10, seed=0)
    with pytest.raises(RuntimeError, match="ask"):
        run.tell([0.0] * 20)
    with pytest.raises(RuntimeError, match="not done"):
        run.result()
    sols = run.ask()
    assert len(sols) == 20 and run.codes.shape == (20, 10)
    with pytest.raises(RuntimeError, match="before asking again"):
        run.ask()
    with pytest.raises(ValueError, match="expected 20"):
        run.tell([0.0] * 19)
    with pytest.raises(ValueError):
        run.tell(["x"] * 20)
    run.tell([sphere_obj(s) for s in sols])  # the run goes on after wrong values
    assert not run.done
    run.tell([sphere_obj(s) for s in run.ask()])
    assert run.done and run.result().nfe == 40
    with pytest.raises(RuntimeError, match="done"):
        run.ask()
    with pytest.raises(RuntimeError, match="done"):
        run.tell([])
    assert run.stop() is run.result()  # stopping a finished run gives its result
    assert no_worker()


def test_vectorized_wrong_shape(sphere10):
    p = ypea.Problem(sphere10.vars, None, vectorized=True)
    run = ypea.PSO(max_nfe=40, pop_size=20).ask_tell(p, seed=0)
    batch = run.ask()
    assert batch.x.shape == (20, 10)
    with pytest.raises(ValueError):
        run.tell(np.zeros(19))
    run.tell(np.sum(batch.x**2, axis=1))
    run.stop()
    assert no_worker()


def test_stop(sphere10):
    run = ypea.PSO(max_nfe=1000, pop_size=20).ask_tell(sphere10, seed=0)
    run.tell([sphere_obj(s) for s in run.ask()])
    run.ask()  # stopping with a batch asked but not told
    r = run.stop("lab closed")
    assert r.stop_reason == "lab closed" and r.nfe == 20 and run.done
    assert no_worker()

    # stopped before any value was told: there is no result
    run = ypea.PSO(max_nfe=1000).ask_tell(sphere10, seed=0)
    with pytest.raises(RuntimeError, match="before any solution"):
        run.stop()
    assert no_worker()


class Broken(ypea.PSO):
    def iterate(self):
        raise ZeroDivisionError("from the algorithm")


def test_errors_reach_the_caller(sphere10):
    def bad(alg):
        raise KeyError("from the callback")

    run = ypea.PSO(max_nfe=1000, pop_size=20).ask_tell(sphere10, seed=0, callbacks=[bad])
    run.tell([sphere_obj(s) for s in run.ask()])  # the initialization
    with pytest.raises(KeyError, match="from the callback"):
        run.tell([sphere_obj(s) for s in run.ask()])  # the first iteration, then the callback
    assert run.done and no_worker()
    with pytest.raises(RuntimeError, match="failed"):
        run.result()

    run = Broken(max_nfe=1000, pop_size=20).ask_tell(sphere10, seed=0)
    with pytest.raises(ZeroDivisionError, match="from the algorithm"):
        run.tell([sphere_obj(s) for s in run.ask()])
    assert run.done and no_worker()


def test_abandoned_run_ends_its_worker(sphere10):
    import gc

    run = ypea.PSO(max_nfe=1000).ask_tell(sphere10, seed=0)
    thread = run._thread
    del run
    gc.collect()
    thread.join(5)
    assert not thread.is_alive()


def test_live_run_is_independent(sphere10):
    alg = ypea.PSO(max_nfe=200, pop_size=20)
    run = alg.ask_tell(sphere10, seed=5)
    expected = alg.solve(sphere10, seed=5)  # the same object solves while the run waits
    drive(run, sphere_obj)
    same(run.result(), expected)


@pytest.mark.parametrize("tells", [1, 4])
@pytest.mark.parametrize("name", NAMES)
def test_stop_every_algorithm(name, tells, sphere10):
    # some algorithms ask for one solution at a time, so a stop can land inside initialize()
    run = make(name, max_nfe=2000).ask_tell(sphere10, seed=0)
    told = 0
    for _ in range(tells):
        sols = run.ask()
        told += len(sols)
        run.tell([sphere_obj(s) for s in sols])
    r = run.stop("halt")
    assert r.stop_reason == "halt" and r.nfe == told
    assert no_worker()
