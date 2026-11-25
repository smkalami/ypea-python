"""The Studio generator: exact parity with YPEA Studio (cases in tests/data/studio)."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_algorithms import NAMES, make

import ypea
from ypea.random import philox

DATA = Path(__file__).parent / "data" / "studio"


def test_philox_test_vectors():
    cases = [
        ((0x00000000, 0x00000000, 0x00000000), (0xFF1DAE59, 0x6CD10DF2)),
        ((0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), (0x2C3F628B, 0xAB4FD7AD)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E), (0xDD7CE038, 0xF62A4C12)),
    ]
    for (c0, c1, key), expected in cases:
        a, b = philox(np.array([c0], np.uint64), np.array([c1], np.uint64), key)
        assert (int(a[0]), int(b[0])) == expected


def test_draws_match_studio_exactly():
    data = json.loads((DATA / "rng.json").read_text())
    rng = ypea.StudioRng(data["seed"])
    for d in data["draws"]:
        m, args, values = d["method"], d["args"], d["values"]
        n = len(values)
        if m == "random":
            got = rng.random(n)
        elif m == "uniform":
            got = rng.uniform(*args, n)
        elif m == "integers":
            got = rng.integers(*args, size=n)
        elif m == "standardNormal":
            got = rng.standard_normal(n)
        elif m == "standardCauchy":
            got = rng.standard_cauchy(n)
        elif m == "geometric":
            got = rng.geometric(args[0], n)
        elif m == "choice":
            got = rng.choice(len(args[0]), size=n, p=args[0])
        elif m == "permutation":
            got = [rng.permutation(args[0])]
        assert np.asarray(got).ravel().tolist() == np.asarray(values).ravel().tolist(), m  # strict
    ds = data["derivedSeeds"]
    assert [ypea.derive_seed(ds["base"], i) for i in range(len(ds["seeds"]))] == ds["seeds"]


def sphere_in_order(sol):
    s = 0.0
    for v in sol.x:  # summed in index order, as the Studio built-in does
        s += v * v
    return s


def test_pso_trace_matches_studio_exactly():
    data = json.loads((DATA / "pso_sphere.json").read_text())
    p = ypea.Problem(ypea.Real("x", shape=10, lb=-10, ub=10), sphere_in_order)
    seed = ypea.StudioRng(ypea.derive_seed(data["project"]["run"]["seed"], 0))
    alg = ypea.PSO(pop_size=30, max_nfe=data["project"]["run"]["budget"])
    for _ in range(2):  # the same StudioRng object gives the same run twice
        r = alg.solve(p, seed=seed)
        assert r.nfe == data["nfe"]
        assert r.history.trace_nfe.tolist() == data["traceNfe"]
        assert r.history.trace_obj_value.tolist() == data["traceF"]
        assert r.best.position.tolist() == data["bestCodes"]
    assert seed.pos == 0  # the caller's object is not consumed


@pytest.mark.parametrize("name", NAMES)
def test_every_algorithm_runs_on_the_studio_generator(sphere10, name):
    # catches any algorithm that calls a Generator method StudioRng does not offer
    alg = make(name, max_nfe=800)
    a = alg.solve(sphere10, seed=ypea.StudioRng(42))
    b = alg.solve(sphere10, seed=ypea.StudioRng(42))
    assert a.nfe == b.nfe == 800 or a.stop_reason == b.stop_reason
    assert a.best.obj_value == b.best.obj_value


def test_non_finite_values_count_as_worst():
    def f(sol):
        return -np.inf if sol.x[0] < 0.5 else float(sol.x[0])

    r = ypea.DE(max_iter=10, pop_size=10).solve(ypea.Problem(ypea.Real("x", shape=2), f), seed=0)
    assert np.isfinite(r.best.obj_value) and r.best.solution.x[0] >= 0.5
