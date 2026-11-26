"""MOEA/D, MOPSO and SPEA2: hand worked steps from the papers, fronts on ZDT1 and DTLZ2, and
the checks every multi-objective algorithm must pass."""

import pickle

import numpy as np
import pytest
from problems import mixed_obj

import ypea
from ypea.algorithms import mopso
from ypea.algorithms.moead import aggregate
from ypea.algorithms.mopso import hypercubes, leader_probabilities, mutate, mutation_probability, select_leaders
from ypea.algorithms.spea2 import spea2_fitness, truncate
from ypea.benchmarks.multi import dtlz2, pareto_front, zdt1
from ypea.indicators import igd
from ypea.operators import NativeMutation, swap

CASES = {
    "MOEAD": lambda **k: ypea.MOEAD(**k),
    "MOEAD-batch": lambda **k: ypea.MOEAD(batch=True, **k),
    "MOPSO": lambda **k: ypea.MOPSO(**k),
    "SPEA2": lambda **k: ypea.SPEA2(**k),
}
NAMES = sorted(CASES)


def make(name, **kw):
    kw.setdefault("pop_size", 40)
    return CASES[name](**kw)


def two_obj(sol):
    x = sol.x
    return np.column_stack([x[:, 0], (1 + x[:, 1]) * (1 - np.sqrt(x[:, 0])) + x[:, 2] ** 2])


TWO = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=1), two_obj, vectorized=True, n_obj=2, name="two")
ZDT1 = ypea.Problem(ypea.Real("x", shape=30, lb=0, ub=1), lambda s: zdt1(s.x), vectorized=True, n_obj=2)
DTLZ2 = ypea.Problem(ypea.Real("x", shape=12, lb=0, ub=1), lambda s: dtlz2(s.x), vectorized=True, n_obj=3)


def same_front(a, b):
    assert a.nfe == b.nfe and a.iterations == b.iterations
    assert a.front.obj_values.tolist() == b.front.obj_values.tolist()
    assert a.front.positions.tolist() == b.front.positions.tolist()


# hand worked steps


def test_tchebycheff_weighted_sum_and_pbi_values():
    F, w, z = [[0.5, 0.2]], [0.3, 0.7], [0.1, 0.0]
    assert aggregate(F, w, z)[0] == pytest.approx(max(0.3 * 0.4, 0.7 * 0.2))  # 0.14
    assert aggregate(F, w, z, "weighted_sum")[0] == pytest.approx(0.3 * 0.5 + 0.7 * 0.2)
    # along w = (1, 0): d1 = 0.4 and d2 = 0.2, so 0.4 + 5 * 0.2
    assert aggregate(F, [1.0, 0.0], z, "pbi", theta=5)[0] == pytest.approx(1.4)
    # a zero weight ignores an infinite (worst) value instead of giving NaN
    assert aggregate([[np.inf, 0.2]], [0.0, 1.0], [0.0, 0.0])[0] == pytest.approx(0.2)
    assert aggregate([[np.nan, 0.2]], [0.5, 0.5], [0.0, 0.0])[0] == np.inf


def test_moead_neighborhoods_and_population():
    alg = ypea.MOEAD(pop_size=100, neighbors=20, max_iter=1)
    alg.solve(DTLZ2, seed=0)
    assert alg.weights.shape == (91, 3) and np.allclose(alg.weights.sum(axis=1), 1)  # H = 12
    assert alg.neighborhood.shape == (91, 20) and (alg.neighborhood[:, 0] == np.arange(91)).all()
    d = np.linalg.norm(alg.weights[:, None] - alg.weights[None], axis=-1)
    far = np.take_along_axis(d, alg.neighborhood, 1).max(axis=1)
    inside = np.zeros_like(d, bool)
    np.put_along_axis(inside, alg.neighborhood, True, 1)
    assert (d[~inside] >= far.repeat(91 - 20)).all()  # the T closest weight vectors
    assert ypea.MOEAD(pop_size=10, neighbors=50, max_iter=2).solve(TWO, seed=0).nfe == 30  # T clamped to 10
    with pytest.raises(ValueError, match="number of objectives"):
        ypea.MOEAD(pop_size=2).solve(DTLZ2)


def test_moead_ideal_point_and_replacement():
    alg = ypea.MOEAD(pop_size=20, max_iter=3)
    r = alg.solve(TWO, seed=2, keep_evaluations=True)
    # the ideal point is the best value of each objective over every evaluation
    assert np.allclose(alg.ideal, r.evaluations.obj_values.min(axis=0))
    # every solution of the population was evaluated, and the population holds copies of
    # good children (replacement of neighbors)
    ev = {tuple(f) for f in r.evaluations.obj_values.tolist()}
    assert all(tuple(f) in ev for f in alg.pop.F.tolist())
    assert len({tuple(f) for f in alg.pop.F.tolist()}) < len(alg.pop)


def test_hypercube_density_roulette():
    F = [[0.0, 0.0], [0.05, 0.95], [0.06, 0.9], [1.0, 1.0]]
    index, counts = hypercubes(F, 2)
    # cells (0, 0), (0, 1), (0, 1) and (1, 1): the top of the range goes to the last cell
    assert index.tolist() == [0, 1, 1, 2] and counts.tolist() == [1, 2, 1]
    # fitness 10 / count = 10, 5, 10
    assert leader_probabilities(counts).tolist() == pytest.approx([0.4, 0.2, 0.4])
    # no range in one objective, and a non-finite value
    index, counts = hypercubes([[0.0, 1.0], [1.0, 1.0], [np.inf, 1.0]], 4)
    assert counts.tolist() == [1, 2] and index.tolist() == [0, 1, 1]  # 1 and inf share the last cell


def test_mopso_leader_choice():
    # members 0 and 4 are alone in their hypercubes, and 1, 2 and 3 share one: fitness 10, 10 / 3 and 10
    index, counts = np.array([0, 1, 2, 1, 1]), np.array([1, 3, 1])
    picks = select_leaders(index, counts, 70_000, np.random.default_rng(0))
    share = np.bincount(picks, minlength=5) / len(picks)
    # the crowded hypercube is chosen with probability 1/7, and then each member a third of the time
    assert share == pytest.approx([3 / 7, 1 / 21, 3 / 7, 1 / 21, 1 / 21], abs=0.005)
    # a member is drawn from the chosen hypercube only: one occupied hypercube gives all of them
    picks = select_leaders(np.zeros(4, int), np.array([4]), 4000, np.random.default_rng(1))
    assert np.bincount(picks).tolist() == pytest.approx([1000] * 4, abs=100)


def test_mopso_mutation_operator():
    X = np.random.default_rng(0).random((2000, 5))
    for p in (1.0, 0.2):
        Y = mutate(X.copy(), p, np.random.default_rng(1))
        changed = Y != X
        assert changed.sum(axis=1).max() == 1  # at most one component per particle
        assert np.mean(changed.any(axis=1)) == pytest.approx(p, abs=0.03)
        rows, cols = np.nonzero(changed)
        # redrawn uniformly within p of the old value, clipped to [0, 1]
        lo, hi = np.maximum(X[rows, cols] - p, 0), np.minimum(X[rows, cols] + p, 1)
        assert ((Y[rows, cols] >= lo) & (Y[rows, cols] <= hi)).all()
        if p == 1.0:
            assert Y[rows, cols].min() < 0.01 and Y[rows, cols].max() > 0.99  # the full range at the start
    assert (mutate(X.copy(), 0.0, np.random.default_rng(1)) == X).all()


def test_mopso_mutation_rate_schedule_over_the_run(monkeypatch):
    # (1 - t) ** (5 / rate), with t the share of the run used before the iteration
    assert mutation_probability(0.0, 0.5) == 1 and mutation_probability(0.5, 0.5) == 0.5**10
    assert mutation_probability(0.5, 1.0) == 0.5**5 and mutation_probability(0.0, 0) == 0
    used = []
    monkeypatch.setattr(mopso, "mutate", lambda X, p, rng: used.append(p) or X)
    ypea.MOPSO(pop_size=10, max_iter=5, mutation_rate=0.5).solve(TWO, seed=0)
    assert used == pytest.approx([(1 - t) ** 10 for t in (0, 0.25, 0.5, 0.75, 1.0)])
    used.clear()
    ypea.MOPSO(pop_size=10, max_nfe=50, mutation_rate=0.25).solve(TWO, seed=0)
    assert used == pytest.approx([(1 - t) ** 20 for t in (0.2, 0.4, 0.6, 0.8)])  # nfe / max_nfe


def diagonal(sol):
    return np.column_stack([sol.x[:, 0], sol.x[:, 0]])


def tradeoff(sol):
    return np.column_stack([sol.x[:, 0], 1 - sol.x[:, 0]])


@pytest.mark.parametrize("objective", [diagonal, tradeoff])
def test_mopso_personal_best(objective):
    p = ypea.Problem(ypea.Real("x", shape=1, lb=0, ub=1), objective, vectorized=True, n_obj=2)
    alg = ypea.MOPSO(pop_size=2000, max_iter=1)
    r = alg.solve(p, seed=0, keep_evaluations=True)
    first = r.evaluations.positions[:2000, 0]
    new, best = alg.pop.X[:, 0], alg.pbest.X[:, 0]
    if objective is diagonal:
        # one solution always dominates the other: the personal best is the smaller x
        assert (best == np.minimum(first, new)).all()
    else:
        # neither dominates: the new position replaces the old one on a coin flip
        moved = new != first
        assert ((best == first) | (best == new)).all()
        assert np.mean(best[moved] == new[moved]) == pytest.approx(0.5, abs=0.03)


def test_mopso_repository_and_mutation_schedule():
    alg = ypea.MOPSO(pop_size=30, repository_size=10, divisions=5, max_nfe=600)
    r = alg.solve(TWO, seed=1)
    rep = alg.repository
    assert len(rep) <= 10
    assert not TWO.comparator.dominates(rep.F[:, None], rep.V[:, None], rep.F[None], rep.V[None]).any()
    assert len(r.front) > len(rep)  # the front of the run is kept apart and is not truncated


def test_moead_batch_evaluates_a_generation_at_once():
    sizes = []

    def counted(sol):
        sizes.append(len(sol.x))
        return two_obj(sol)

    p = ypea.Problem(TWO.vars, counted, vectorized=True, n_obj=2)
    r = ypea.MOEAD(pop_size=20, max_nfe=130, batch=True).solve(p, seed=0)
    # the initial population, then one call per generation, and the last one cut at max_nfe
    assert sizes == [20] * 6 + [10] and r.nfe == 130
    sizes.clear()
    ypea.MOEAD(pop_size=20, max_nfe=130).solve(p, seed=0)
    assert sizes == [20] + [1] * 110  # the paper: one child at a time
    with pytest.raises(ValueError, match="batch"):
        ypea.MOEAD(batch=1)


def test_spea2_fitness_by_hand():
    # minimized: A dominates D, B dominates C and D, C dominates D
    F = np.array([[1.0, 4.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    c = ypea.comparison.Comparator(["min", "min"])
    dom = c.dominates(F[:, None], 0, F[None], 0)
    # S = 1, 2, 1, 0 and R = 0, 0, S(B) = 2, S(A) + S(B) + S(C) = 4
    sigma = np.array([np.sqrt(5), np.sqrt(2), np.sqrt(2), np.sqrt(2)])  # nearest neighbor, k = 1
    assert spea2_fitness(dom, F, k=1) == pytest.approx([0, 0, 2, 4] + 1 / (sigma + 2))
    # k = floor(sqrt(4)) = 2 by default: second nearest
    sigma2 = np.array([np.sqrt(5), np.sqrt(5), np.sqrt(2), np.sqrt(8)])
    assert spea2_fitness(dom, F) == pytest.approx([0, 0, 2, 4] + 1 / (sigma2 + 2))


def test_spea2_truncation_by_hand():
    F = np.array([[0.0, 0.0], [1.0, 0.0], [1.1, 0.0], [3.0, 0.0]])
    # 1 and 1.1 are both 0.1 from their nearest neighbor. The second nearest decides: 1 is
    # 1.0 from 0, and 1.1 is 1.1 from 0, so 1 goes. Then 0 and 1.1 are both 1.1 apart, and 1.1 goes
    # (its second distance 1.9 is below the 3 of point 0), and the boundary points stay.
    assert truncate(F, 3).tolist() == [0, 2, 3]
    assert truncate(F, 2).tolist() == [0, 3]
    assert truncate(F, 4).tolist() == [0, 1, 2, 3]


def test_spea2_archive_and_environmental_selection():
    alg = ypea.SPEA2(pop_size=20, archive_size=10, max_iter=5)
    alg.solve(TWO, seed=0)
    assert len(alg.archive) == 10 and (alg.fitness < 1).all()  # a full non-dominated archive
    small = ypea.SPEA2(pop_size=6, archive_size=30, max_iter=1)
    small.solve(TWO, seed=0)
    assert len(small.archive) == 12  # fewer solutions than the archive size: all are kept
    assert np.all(np.diff(small.fitness) >= 0)  # filled by fitness


# fronts on the papers' test problems


# IGD at 10 000 evaluations with seed 0 (reached: MOEA/D 0.022 and 0.017, MOPSO 0.56 and 0.084,
# SPEA2 0.027 and 0.023). Random search reaches 1.6 and 0.24. MOPSO converges slowly on the 30
# variables of ZDT1 (0.32 at 25 000 evaluations).
IGD = {
    ("MOEAD", "zdt1"): 0.04, ("MOPSO", "zdt1"): 0.75, ("SPEA2", "zdt1"): 0.05,
    ("MOEAD", "dtlz2"): 0.03, ("MOPSO", "dtlz2"): 0.12, ("SPEA2", "dtlz2"): 0.04,
    ("MOEAD-batch", "zdt1"): 0.04, ("MOEAD-batch", "dtlz2"): 0.03,
}


@pytest.mark.parametrize("name", NAMES)
def test_igd_on_zdt1_and_dtlz2(name):
    r = CASES[name](pop_size=100, max_nfe=10_000).solve(ZDT1, seed=0)
    assert igd(r.front.obj_values, pareto_front("zdt1", 500)) < IGD[name, "zdt1"]
    r = CASES[name](pop_size=91, max_nfe=10_000).solve(DTLZ2, seed=0)
    assert igd(r.front.obj_values, pareto_front("dtlz2", 990, 3)) < IGD[name, "dtlz2"]


# the checks of every algorithm


@pytest.mark.parametrize("name", NAMES)
def test_senses_give_mirrored_runs(name):
    r = make(name, max_nfe=800).solve(TWO, seed=3)
    neg = ypea.Problem(TWO.vars, lambda s: -two_obj(s), vectorized=True, sense="max", n_obj=2)
    m = make(name, max_nfe=800).solve(neg, seed=3)
    assert m.front.obj_values.tolist() == (-r.front.obj_values).tolist()
    mixed = ypea.Problem(TWO.vars, lambda s: two_obj(s) * [1, -1], vectorized=True, sense=["min", "max"])
    m = make(name, max_nfe=800).solve(mixed, seed=3)
    assert m.front.obj_values.tolist() == (r.front.obj_values * [1, -1]).tolist()


@pytest.mark.parametrize("constraints", ["feasibility", "penalty"])
@pytest.mark.parametrize("name", NAMES)
def test_constrained(name, constraints):
    def f(sol):
        return two_obj(sol), 0.5 - sol.x[:, 0]  # feasible when x0 >= 0.5

    p = ypea.Problem(TWO.vars, f, vectorized=True, n_obj=2, constraints=constraints, penalty=100)
    r = make(name, max_nfe=1500).solve(p, seed=0)
    assert r.front.feasible.all() and (r.front.obj_values[:, 0] >= 0.5).all()
    assert len(r.front) >= 5


@pytest.mark.parametrize("name", NAMES)
def test_budget_reproducibility_and_pickling(name):
    alg = make(name, max_nfe=437)
    a = alg.solve(TWO, seed=5)
    assert a.nfe == 437 and a.stop_reason == "max_nfe"
    same_front(a, alg.solve(TWO, seed=5))
    same_front(a, pickle.loads(pickle.dumps(alg)).solve(TWO, seed=5))
    assert len(make(name, max_nfe=600, max_archive=6).solve(TWO, seed=5).front) <= 6


@pytest.mark.parametrize("name", NAMES)
def test_studio_generator(name):
    alg = make(name, max_nfe=500)
    seed = ypea.StudioRng(11)
    same_front(alg.solve(TWO, seed=seed), alg.solve(TWO, seed=seed))


@pytest.mark.parametrize("name", NAMES)
def test_every_variable_type(name, mixed):
    p = ypea.Problem(mixed.vars, lambda s: [mixed_obj(s), float(np.sum(s.p[:2]) - np.sum(s.r))], n_obj=2)
    r = make(name, max_nfe=400, pop_size=20).solve(p, seed=0)
    assert r.nfe == 400 and len(r.front) >= 1
    s = r.front.solutions[0]
    assert sorted(s.p.tolist()) == list(range(5)) and len(s.q) == 3


@pytest.mark.parametrize("name", NAMES)
def test_ask_tell_matches_solve(name):
    alg = make(name, max_nfe=300)
    run = alg.ask_tell(TWO, seed=4)
    while not run.done:
        run.tell(two_obj(run.ask()))
    same_front(run.result(), alg.solve(TWO, seed=4))


@pytest.mark.parametrize("name", NAMES)
def test_wrong_use(name):
    with pytest.raises(ValueError, match="multi-objective"):
        make(name, max_nfe=100).solve(ypea.Problem(ypea.Real("x", shape=2), lambda s: float(s.x.sum())))
    with pytest.raises(ValueError, match="max_archive"):
        make(name, max_archive=0)
    if name != "MOPSO":
        with pytest.raises(ValueError, match="native"):
            make(name, mutation=NativeMutation(swap, ypea.Permutation))
        with pytest.raises(TypeError, match="crossover"):
            make(name, crossover="sbx")
