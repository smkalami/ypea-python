"""Checks that target one algorithm's own behavior."""

import numpy as np

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
    assert np.isclose(alg.step(), alg.final_step_size + (alg.step_size - alg.final_step_size) * (1 / 30) ** 2)


def test_iwo_auto_fitness_switches_to_ranks_with_constraints(constrained, sphere10):
    a = ypea.IWO(max_iter=2)
    a.solve(sphere10, seed=0)
    assert not a.constrained
    a.solve(constrained, seed=0)
    assert a.constrained
