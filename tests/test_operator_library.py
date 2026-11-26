"""The operator library: SBX, polynomial mutation, native operators and compatibility checks."""

import numpy as np
import pytest

import ypea
from ypea.operators import SBX, NativeCrossover, NativeMutation, PolynomialMutation, uniform_crossover

STUDIO_METHODS = ("random", "uniform", "integers", "standard_normal", "standard_cauchy", "geometric",
                  "choice", "permutation")


class Spy:
    """Passes only the Generator methods that StudioRng offers, and records the calls."""

    def __init__(self, rng):
        self.rng, self.calls = rng, []

    def __getattr__(self, name):
        if name not in STUDIO_METHODS:
            raise AttributeError(f"{name} is not a StudioRng method")
        self.calls.append(name)
        return getattr(self.rng, name)


def parents(n=400, d=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, d)), rng.random((n, d))


@pytest.mark.parametrize("eta", [0.5, 2, 20, 100])
def test_sbx_bounds_and_mean(eta):
    x1, x2 = parents()
    x1[:5], x2[:5] = 0.0, 1.0  # parents at the bounds
    x2[5:10] = x1[5:10]  # equal parents are copied
    y1, y2 = SBX(eta=eta, prob=1.0, prob_var=1.0)(x1, x2, np.random.default_rng(1))
    assert y1.min() >= 0 and y2.max() <= 1 and y1.shape == x1.shape
    assert np.array_equal(y1[5:10], x1[5:10]) and np.array_equal(y2[5:10], x2[5:10])
    # the bounded form spreads the children around the parents' midpoint, never past the bounds
    lo, hi = np.minimum(x1, x2), np.maximum(x1, x2)
    mid = (lo + hi) / 2
    assert np.all((np.minimum(y1, y2) <= mid + 1e-12) & (np.maximum(y1, y2) >= mid - 1e-12))


def test_sbx_spread_shrinks_with_eta():
    x1, x2 = np.full((20000, 1), 0.4), np.full((20000, 1), 0.6)
    spread = [np.mean(np.abs(SBX(eta=e, prob=1, prob_var=1)(x1, x2, np.random.default_rng(0))[0] - 0.5))
              for e in (1, 5, 20, 100)]
    assert spread == sorted(spread, reverse=True)
    # beta < 1 (children inside the parents) half of the time for the unbounded case
    y = SBX(eta=20, prob=1, prob_var=1)(x1, x2, np.random.default_rng(0))[0]
    assert 0.45 < np.mean(np.abs(y - 0.5) < 0.1) < 0.6


def test_sbx_probabilities():
    x1, x2 = parents(2000, 10)
    y1, _ = SBX(prob=0.0)(x1, x2, np.random.default_rng(0))
    assert np.array_equal(y1, x1)
    y1, _ = SBX(prob=1.0, prob_var=0.5)(x1, x2, np.random.default_rng(0))
    assert 0.45 < np.mean(y1 != x1) < 0.55


def test_polynomial_mutation():
    x = np.random.default_rng(0).random((5000, 10))
    x[:10] = 0.0
    x[10:20] = 1.0
    y = PolynomialMutation(prob=1.0)(x, np.random.default_rng(1))
    assert y.min() >= 0 and y.max() <= 1 and np.mean(y != x) > 0.99
    assert 0.08 < np.mean(PolynomialMutation()(x, np.random.default_rng(2)) != x) < 0.12  # 1/d
    small = np.mean(np.abs(PolynomialMutation(eta=100, prob=1)(x, np.random.default_rng(3)) - x))
    large = np.mean(np.abs(PolynomialMutation(eta=1, prob=1)(x, np.random.default_rng(3)) - x))
    assert small < large
    # away from the bounds the change is symmetric around 0
    mid = np.full((20000, 1), 0.5)
    assert abs(np.mean(PolynomialMutation(prob=1)(mid, np.random.default_rng(4)) - 0.5)) < 2e-3


@pytest.mark.parametrize("op", [SBX(), PolynomialMutation()])
def test_operators_use_only_studio_draws(op):
    x1, x2 = parents(30, 4)
    args = (x1, x2) if isinstance(op, SBX) else (x1,)
    spy = Spy(np.random.default_rng(0))
    op(*args, spy)
    assert spy.calls and set(spy.calls) <= {"random"}
    a = op(*args, ypea.StudioRng(9))
    b = op(*args, ypea.StudioRng(9))
    assert np.array_equal(np.asarray(a), np.asarray(b))
    # the draws have fixed shapes, so they do not depend on the values
    spy2 = Spy(np.random.default_rng(0))
    op(*(np.zeros_like(x) for x in args), spy2)
    assert spy2.calls == spy.calls


def sphere(sol):
    return float(np.sum(sol.x**2))


def test_ga_and_sa_with_library_operators():
    p = ypea.Problem(ypea.Real("x", shape=5, lb=-5, ub=5), sphere)
    r = ypea.GA(max_nfe=4000, pop_size=40, crossover=SBX(), mutation=PolynomialMutation()).solve(p, seed=0)
    assert r.best.obj_value < 0.05
    ref = ypea.GA(max_nfe=4000, pop_size=40).solve(p, seed=0)
    assert r.best.obj_value != ref.best.obj_value
    r = ypea.SA(max_nfe=4000, mutation=PolynomialMutation(prob=1)).solve(p, seed=0)
    assert r.best.obj_value < 1
    assert r.nfe == 4000
    with pytest.raises(TypeError):
        ypea.SA(crossover=SBX())


def test_native_library_operators():
    def f(sol):
        return float(np.sum(sol.i) + np.sum(sol.x**2))

    p = ypea.Problem([ypea.Integer("i", shape=4, lb=0, ub=9), ypea.Real("x", shape=2, lb=-1, ub=1)], f)
    calls = []

    def flip(x, rng):
        calls.append(x.shape)
        return np.where(rng.random(x.shape) < 0.5, 0, x)

    ga = ypea.GA(max_nfe=600, space="native", crossover=NativeCrossover(uniform_crossover, types=ypea.Integer),
                 mutation=NativeMutation(flip, types=[ypea.Integer]))
    r = ga.solve(p, seed=0)
    assert calls and all(s[1:] == (4,) for s in calls) and r.best.obj_value < 3
    ypea.SA(max_nfe=300, space="native", mutation=NativeMutation(flip, types=ypea.Integer)).solve(p, seed=0)


def test_operator_compatibility_errors():
    p = ypea.Problem(ypea.Real("x", shape=2, lb=0, ub=1), sphere)
    with pytest.raises(ValueError, match="space='encoded'"):
        ypea.GA(space="native", crossover=SBX())
    with pytest.raises(ValueError, match="space='native'"):
        ypea.GA(mutation=NativeMutation(lambda x, rng: x, types=ypea.Integer))
    with pytest.raises(TypeError, match="Crossover"):
        ypea.GA(crossover=PolynomialMutation())
    with pytest.raises(TypeError, match="Mutation"):
        ypea.GA(mutation=SBX())
    with pytest.raises(TypeError):
        ypea.GA(crossover=lambda a, b, rng: (a, b))
    ga = ypea.GA(max_nfe=100, space="native", crossover=NativeCrossover(uniform_crossover, types=ypea.Permutation))
    with pytest.raises(ValueError, match="Permutation"):
        ga.solve(p)
    with pytest.raises(ValueError, match="Permutation"):
        ga.ask_tell(p)
    with pytest.raises(ValueError, match="eta"):
        SBX(eta=-1)
    with pytest.raises(ValueError, match="prob"):
        PolynomialMutation(prob=2)
