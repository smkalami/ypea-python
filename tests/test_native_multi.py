"""space="native" for NSGA-II, NSGA-III, SPEA2 and MOEA/D."""

import numpy as np
import pytest

import ypea
from ypea import benchmarks as B
from ypea.operators import SBX, NativeCrossover, uniform_crossover

ALGS = [ypea.NSGAII, ypea.NSGAIII, ypea.SPEA2, ypea.MOEAD]


def tsp2(n=12, seed=5):
    """A TSP with two distance matrices, so two objectives."""
    rng = np.random.default_rng(seed)
    d = [np.linalg.norm(c[:, None] - c[None], axis=-1) for c in rng.random((2, n, 2))]

    def objective(s):
        t = s.tour
        nxt = np.roll(t, -1, axis=-1)
        return np.stack([d[0][t, nxt].sum(-1), d[1][t, nxt].sum(-1)], axis=-1)
    return ypea.Problem(ypea.Permutation("tour", n), objective, vectorized=True, n_obj=2, name="tsp2")


def mixed():
    """A permutation, an integer and a real variable."""
    def objective(s):
        cost = np.abs(np.diff(s.order)).sum() + s.k.sum()
        return [cost + s.x[0] ** 2, 10 - s.k.sum() + (s.x[0] - 1) ** 2]
    return ypea.Problem([ypea.Permutation("order", 6), ypea.Integer("k", 3, 0, 4), ypea.Real("x", 1, -2, 2)],
                        objective, n_obj=2, name="mixed")


@pytest.mark.parametrize("cls", ALGS)
def test_only_real_variables_run_as_encoded(cls):
    p = B.multi_problem("zdt1", dim=6, front_points=50)
    a = cls(max_nfe=1500).solve(p, seed=3)
    b = cls(max_nfe=1500, space="native").solve(p, seed=3)
    assert np.array_equal(a.front.obj_values, b.front.obj_values)


@pytest.mark.parametrize("cls", ALGS)
def test_native_keeps_every_type_valid_and_is_seeded(cls):
    p = mixed()
    seen = []
    r = cls(max_nfe=800, pop_size=20, space="native").solve(
        p, seed=2, callbacks=[lambda a: seen.append(p.decode(a.pop.X))])
    for sols in seen:
        for s in sols:
            assert sorted(s.order) == list(range(6)) and np.all((s.k >= 0) & (s.k <= 4))
    again = cls(max_nfe=800, pop_size=20, space="native").solve(p, seed=2)
    assert np.array_equal(r.front.obj_values, again.front.obj_values) and r.nfe == 800


def test_native_helps_on_a_two_objective_tsp():
    # the shortest tour for each objective, added up over three seeds
    p = tsp2(20)
    total = {s: sum(ypea.NSGAII(max_nfe=20_000, space=s).solve(p, seed=k).front.obj_values.min(axis=0).sum()
                    for k in range(3)) for s in ("encoded", "native")}
    assert total["native"] < 0.95 * total["encoded"]


def test_native_operators_and_settings_are_checked():
    p = tsp2()
    with pytest.raises(ValueError, match="space='encoded'"):
        ypea.NSGAII(space="native", crossover=SBX()).solve(p, seed=0)
    with pytest.raises(ValueError, match="space='native'"):
        ypea.SPEA2(crossover=NativeCrossover(uniform_crossover, types=ypea.Permutation)).solve(p, seed=0)
    with pytest.raises(ValueError, match="operators need space='native'"):
        ypea.MOEAD(operators={"tour": None}).solve(p, seed=0)
    with pytest.raises(ValueError, match="space must be"):
        ypea.NSGAIII(space="codes").solve(p, seed=0)


def test_user_native_operator_is_used():
    calls = []

    def cx(a, b, rng):
        calls.append(1)
        return a.copy(), b.copy()
    op = NativeCrossover(cx, types=ypea.Permutation)
    ypea.NSGAII(max_nfe=200, pop_size=20, space="native", crossover=op).solve(tsp2(), seed=0)
    assert calls


def test_ask_tell_and_studio_rng_with_native_space():
    p = tsp2()
    want = ypea.MOEAD(max_nfe=300, pop_size=20, space="native").solve(p, seed=4)
    run = ypea.MOEAD(max_nfe=300, pop_size=20, space="native").ask_tell(p, seed=4)
    while not run.done:
        run.tell(p.objective(run.ask()))
    assert np.array_equal(run.result().front.obj_values, want.front.obj_values)
    ypea.NSGAII(max_nfe=300, pop_size=20, space="native").solve(p, seed=ypea.StudioRng(1))
