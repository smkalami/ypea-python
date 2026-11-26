import itertools
import pickle

import numpy as np
import pytest

import ypea
from ypea.problems.engineering import (
    gear_train,
    pressure_vessel,
    speed_reducer,
    spring,
    three_bar_truss,
    welded_beam,
)

# (factory, published best value, digits it is published with)
CASES = [
    (pressure_vessel, 6059.714335, 6),
    (lambda: pressure_vessel(mixed=False), 5885.3328, 4),
    (welded_beam, 1.724852, 6),
    (spring, 0.012665233, 9),
    (speed_reducer, 2994.471066, 6),
    (three_bar_truss, 263.8958434, 7),
    (gear_train, 2.700857e-12, 18),
]
IDS = ["pressure_vessel", "pressure_vessel_continuous", "welded_beam", "spring", "speed_reducer",
       "three_bar_truss", "gear_train"]


def _x_opt(p):
    x = p.info["x_opt"]
    return x if isinstance(x, dict) else {p.vars[0].name: x}


@pytest.mark.parametrize(("make", "published", "digits"), CASES, ids=IDS)
def test_best_known_solution(make, published, digits):
    p = make()
    sol = _x_opt(p)
    f, v = p.evaluate(ypea.problem.Solution(**sol))
    assert v <= 1e-9
    assert f == pytest.approx(p.optimum, rel=1e-12)
    assert round(f, digits) == pytest.approx(published, abs=10.0**-digits)
    # the same through the encoding and the vectorized path
    F, V = ypea.problem.parse_batch(p.objective(p.decode(p.encode(sol))), 1)
    assert F[0] == pytest.approx(f, rel=1e-9) and V[0] <= 1e-6


@pytest.mark.parametrize("make", [c[0] for c in CASES], ids=IDS)
def test_vectorized_matches_single_and_pickles(make):
    p = make()
    X = np.random.default_rng(1).random((20, p.code_size))
    F, V = ypea.problem.parse_batch(p.objective(p.decode(X)), 20)
    for i in range(20):
        f, v = p.evaluate(p.decode_one(X[i]))
        assert f == pytest.approx(F[i]) and v == pytest.approx(V[i])
    q = pickle.loads(pickle.dumps(p))
    assert np.allclose(ypea.problem.parse_batch(q.objective(q.decode(X)), 20)[0], F)


def test_bounds_give_no_warnings():
    # corners of the box, where denominators can reach 0
    for make in [c[0] for c in CASES]:
        p = make()
        X = np.array(list(itertools.product([0.0, 1.0], repeat=min(p.code_size, 7))))
        X = np.hstack([X, np.zeros((len(X), p.code_size - X.shape[1]))])
        F, V = ypea.problem.parse_batch(p.objective(p.decode(X)), len(X))
        assert not np.isnan(V).any()


def test_gear_train_optimum_by_enumeration():
    t = np.arange(12, 61, dtype=float)
    ab = (t[:, None] * t[None, :]).ravel()
    best = min(((1 / 6.931 - a / ab) ** 2).min() for a in ab)
    assert best == pytest.approx(gear_train().optimum, rel=1e-12)


def test_pressure_vessel_mixed_thicknesses():
    p = pressure_vessel()
    sol = p.decode_one(p.encode(p.info["x_opt"]))
    assert np.array_equal(sol.thickness, [13, 7])  # 0.8125 and 0.4375 inch


@pytest.mark.parametrize("make", [c[0] for c in CASES[:-1]], ids=IDS[:-1])
def test_lshade_reaches_one_percent(make):
    p = make()
    r = ypea.LSHADE(max_nfe=20_000).solve(p, seed=0)
    assert r.best.feasible
    assert r.best.obj_value <= p.optimum * 1.01


def test_lshade_gear_train_best_of_five():
    # within 1% of 2.7e-12 means the exact optimum, which LSHADE finds on about half the seeds
    p = gear_train()
    best = min(ypea.LSHADE(max_nfe=10_000).solve(p, seed=s).best.obj_value for s in range(5))
    assert best <= p.optimum * 1.01


@pytest.mark.slow
@pytest.mark.parametrize("make", [c[0] for c in CASES[:-1]], ids=IDS[:-1])
def test_lshade_reaches_one_percent_more_seeds(make):
    p = make()
    for seed in range(1, 6):
        r = ypea.LSHADE(max_nfe=20_000).solve(p, seed=seed)
        assert r.best.feasible and r.best.obj_value <= p.optimum * 1.01
