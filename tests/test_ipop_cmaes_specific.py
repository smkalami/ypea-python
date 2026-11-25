"""Checks for IPOP-CMA-ES restarts."""

import numpy as np
import pytest

import ypea
from ypea.algorithms.ipop_cmaes import IPOPCMAES
from ypea.benchmarks import rastrigin


def rastrigin_obj(sol):
    return rastrigin(sol.x)


RASTRIGIN = ypea.Problem(ypea.Real("x", shape=10, lb=-5.12, ub=5.12), rastrigin_obj)


@pytest.mark.slow
def test_restarts_double_the_population_on_rastrigin():
    alg = IPOPCMAES(max_nfe=100_000)
    r = alg.solve(RASTRIGIN, seed=0)
    assert r.nfe == 100_000
    assert alg.restarts >= 3 and len(alg.restart_log) == alg.restarts
    lams = [lam for _, lam, _ in alg.restart_log]
    assert lams == [alg.restart_log[0][1] * 2**k for k in range(alg.restarts)]
    assert lams[0] == 2 * (4 + int(3 * np.log(10))) and alg.lam_current == lams[-1]
    nfes = [nfe for nfe, _, _ in alg.restart_log]
    assert nfes == sorted(nfes) and nfes[-1] < 100_000


@pytest.mark.slow
def test_beats_plain_cmaes_on_rastrigin():
    ipop = [IPOPCMAES(max_nfe=50_000).solve(RASTRIGIN, seed=s).best.obj_value for s in range(3)]
    plain = [ypea.CMAES(max_nfe=50_000).solve(RASTRIGIN, seed=s).best.obj_value for s in range(3)]
    assert np.mean(ipop) < 0.5 * np.mean(plain)


def test_max_restarts_stops_the_run(sphere10):
    alg = IPOPCMAES(max_nfe=200_000, max_restarts=0)
    r = alg.solve(sphere10, seed=0)
    assert alg.restarts == 0 and r.stop_reason == "max_restarts" and r.nfe < 200_000


def flat_obj(sol):
    return 1.0


def test_flat_objective_does_not_blow_up_lambda(sphere10):
    flat = ypea.Problem(sphere10.vars, flat_obj)
    alg = IPOPCMAES(max_iter=60)
    alg.solve(flat, seed=0)
    assert alg.restarts <= 60 // 11
