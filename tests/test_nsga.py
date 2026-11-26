"""NSGA-II and NSGA-III: checks against the papers (Deb et al. 2002, Deb and Jain 2014)."""

import pickle

import numpy as np
import pytest
from problems import mixed_obj

import ypea
from ypea.algorithms.nsga import associate, crowded_tournament, niching, normalize, reference_points
from ypea.benchmarks import dtlz2, pareto_front, zdt1
from ypea.indicators import igd
from ypea.population import Population

ALGS = [ypea.NSGAII, ypea.NSGAIII]


def zdt1_problem(sense="min"):
    sign = 1 if sense == "min" else -1
    return ypea.Problem(ypea.Real("x", shape=30), lambda s: sign * zdt1(s.x), n_obj=2, sense=sense,
                        vectorized=True, name="zdt1")


def dtlz2_problem():
    return ypea.Problem(ypea.Real("x", shape=12), lambda s: dtlz2(s.x, 3), n_obj=3, vectorized=True,
                        name="dtlz2")


class LastPopulation:
    def __call__(self, alg):
        return False

    def on_end(self, alg, result):
        self.F = alg.pop.F.copy()


# convergence: IGD of the front and of the final population (the measure of the papers)

@pytest.fixture(scope="module")
def runs():
    """One run of each algorithm on ZDT1 (10 000 evaluations) and DTLZ2 with 3 objectives
    (92 solutions, 100 iterations): the result and the final population."""
    out = {}
    for cls in ALGS:
        for name, p, kw in (("zdt1", zdt1_problem(), {"max_nfe": 10_000}),
                            ("dtlz2", dtlz2_problem(), {"max_nfe": 9_200, "pop_size": 92})):
            cb = LastPopulation()
            out[cls, name] = cls(**kw).solve(p, seed=0, callbacks=[cb]), cb.F
    return out


@pytest.mark.parametrize("cls", ALGS)
def test_zdt1_igd(runs, cls):
    r, last = runs[cls, "zdt1"]
    front = pareto_front("zdt1", 1000)
    assert igd(r.front.obj_values, front) < 4e-2  # seeds 0 to 2: at most 1.4e-2 (NSGA-II), 2.5e-2
    assert igd(last, front) < 4e-2


@pytest.mark.parametrize("cls", ALGS)
def test_dtlz2_igd(runs, cls):
    r, _ = runs[cls, "dtlz2"]
    assert igd(r.front.obj_values, pareto_front("dtlz2", 1000, 3)) < 4e-2  # seeds 0 to 2: at most 2.3e-2


def test_nsga3_finds_the_targeted_points_of_dtlz2(runs):
    # the paper measures IGD against the points of the front on the reference lines. There
    # NSGA-III is far closer than NSGA-II (seeds 0 to 2: about 1.1e-2 against 7e-2 to 9e-2)
    w = reference_points(3)
    targets = w / np.linalg.norm(w, axis=1, keepdims=True)
    nsga3, nsga2 = igd(runs[ypea.NSGAIII, "dtlz2"][1], targets), igd(runs[ypea.NSGAII, "dtlz2"][1], targets)
    assert nsga3 < 2e-2 and nsga3 < nsga2 / 3


# NSGA-II steps, worked by hand

def test_nsga2_survival_cuts_the_last_front_by_crowding():
    alg = ypea.NSGAII(pop_size=3)
    alg.problem, alg._n = zdt1_problem(), 3
    # front 0: (0, 4), (1, 2), (3, 1), (4, 0). Crowding: the ends are infinite,
    # (1, 2): 3/4 + 3/4 = 1.5 and (3, 1): 3/4 + 2/4 = 1.25. Front 1: (2, 5)
    F = np.array([[3, 1], [2, 5], [0, 4], [1, 2], [4, 0]], float)
    merged = Population(np.zeros((5, 30)), F, np.zeros(5))
    alg.survive(merged)
    assert alg.pop.F.tolist() == [[0, 4], [4, 0], [1, 2]]
    assert alg._rank.tolist() == [0, 0, 0]
    assert alg._crowd.tolist() == [np.inf, np.inf, 1.5]


def test_crowded_tournament():
    rank = np.array([0, 1, 0, 0])
    crowd = np.array([1.0, np.inf, 2.0, 2.0])
    rng = np.random.default_rng(0)
    winners = crowded_tournament(rng, rank, crowd, 2000)
    # the lower front wins, then the larger distance. Of the 16 pairs, 1 wins only (1, 1),
    # 0 wins (0, 0), (0, 1) and (1, 0), and 2 and 3 share the other 12 (their tie by a coin)
    counts = np.bincount(winners, minlength=4) / 2000
    assert counts == pytest.approx([3 / 16, 1 / 16, 6 / 16, 6 / 16], abs=0.04)


# NSGA-III steps

@pytest.mark.parametrize("m, h, n", [(3, 91, 92), (5, 210, 212), (8, 156, 156), (10, 275, 276), (15, 135, 136)])
def test_reference_points_and_population_size_of_table_1(m, h, n):
    ref = reference_points(m)
    assert ref.shape == (h, m) and np.allclose(ref.sum(axis=1), 1)
    assert len(np.unique(ref.round(12), axis=0)) == h
    p = ypea.Problem(ypea.Real("x", shape=m + 4), lambda s: dtlz2(s.x, m), n_obj=m, vectorized=True)
    alg = ypea.NSGAIII(max_iter=1)
    alg.solve(p, seed=0)
    assert len(alg.pop) == n


def test_two_layer_reference_points():
    # Fig. 4 of the paper: p = 2 on the boundary (6 points), p = 1 inside (3 points, shrunk by half)
    ref = reference_points(3, (2, 1))
    assert len(ref) == 9
    inside = ref[6:]
    assert np.allclose(np.sort(inside, axis=1), [[1 / 6, 1 / 6, 2 / 3]] * 3)


def test_normalize_regular_hyperplane():
    # extreme points on the axes at 2, 3 and 4 above the ideal point (1, 1, 1)
    f = np.array([[3, 1, 1], [1, 4, 1], [1, 1, 5], [2, 2, 2]], float)
    fn, ideal, extremes, a = normalize(f)
    assert ideal.tolist() == [1, 1, 1]
    assert np.allclose(a, [2, 3, 4])
    assert np.allclose(fn[:3], np.eye(3))
    assert np.allclose(fn[3], [0.5, 1 / 3, 0.25])


def test_normalize_degenerate_hyperplane():
    # every point has f3 = 0.5, so the extreme points span no plane: the intercepts fall back
    # to the largest translated values of the first front, and a zero range to 1e-12
    f = np.array([[0, 1, 0.5], [1, 0, 0.5], [0.5, 0.5, 0.5]])
    fn, ideal, extremes, a = normalize(f, first=np.array([True, True, True]))
    assert ideal.tolist() == [0, 0, 0.5]
    assert np.allclose(a, [1, 1, 1e-12])
    assert np.all(np.isfinite(fn)) and np.allclose(fn, [[0, 1, 0], [1, 0, 0], [0.5, 0.5, 0]])


def test_normalize_negative_intercept_falls_back_to_the_front():
    # the plane through the extreme points (translated: (1, 0, 1), (0, 2, 0), (1, 1, 3)) has
    # the intercepts (0.8, 2, -4), so each intercept is the largest translated value of the first front
    f = np.array([[3, 2, 2], [1, 3, 0], [2, 2, 3], [2, 1, 1]], float)
    first = np.array([True, True, True, False])
    _, ideal, extremes, a = normalize(f, first=first)
    assert ideal.tolist() == [1, 1, 0]
    assert extremes.tolist() == [[2, 1, 1], [1, 3, 0], [2, 2, 3]]
    assert np.allclose(1 / np.linalg.solve(extremes - ideal, np.ones(3)), [0.8, 2, -4])
    assert a.tolist() == [2, 2, 3]


def test_normalize_keeps_ideal_and_extremes_over_the_run():
    f1 = np.array([[0, 2], [2, 0]], float)
    _, ideal, extremes, _ = normalize(f1)
    f2 = np.array([[1, 1], [1.5, 0.5]])
    fn, ideal, extremes, a = normalize(f2, ideal, extremes)
    assert ideal.tolist() == [0, 0] and np.allclose(a, [2, 2])
    assert np.allclose(fn, f2 / 2)


def test_association_and_niching_by_hand():
    ref = np.array([[1, 0], [0.5, 0.5], [0, 1]], float)
    # p is already selected (line 2). In the last front: a and b are nearest line 0 (at 0.1
    # and 0.3), c nearest line 1 (0.1 / sqrt 2), d nearest line 2 (0.1)
    fn = np.array([[0, 1], [1, 0.1], [0.8, 0.3], [0.5, 0.6], [0.1, 0.95]])
    pi, d = associate(fn, ref)
    assert pi.tolist() == [2, 0, 0, 1, 2]
    assert np.allclose(d, [0, 0.1, 0.3, 0.1 / np.sqrt(2), 0.1])
    last = np.array([False, True, True, True, True])
    rho = np.bincount(pi[~last], minlength=3)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        # lines 0 and 1 are empty, so they get their closest members first, whatever the order
        assert set(niching(2, pi, d, last, rho, rng).tolist()) == {1, 3}
        third = niching(3, pi, d, last, rho, np.random.default_rng(seed))[2]
        assert third in (2, 4)  # line 1 has no member left and is dropped
        assert sorted(niching(4, pi, d, last, rho, np.random.default_rng(seed)).tolist()) == [1, 2, 3, 4]


def test_association_of_non_finite_rows():
    pi, d = associate(np.array([[np.nan, np.nan], [0.2, 0.2]]), np.array([[1, 0], [0, 1.0]]))
    assert d[0] == np.inf and np.isfinite(d[1])


# settings and the framework

@pytest.mark.parametrize("cls", ALGS)
def test_default_operators_follow_the_papers(cls):
    alg = cls(max_iter=1)
    alg.solve(zdt1_problem(), seed=0)
    eta = 20.0 if cls is ypea.NSGAII else 30.0
    prob = 0.9 if cls is ypea.NSGAII else 1.0
    assert (alg._cx.eta, alg._cx.prob, alg._mut.eta, alg._mut.prob) == (eta, prob, 20.0, None)


@pytest.mark.parametrize("cls", ALGS)
def test_settings_are_checked(cls):
    with pytest.raises(ValueError, match="code operator"):
        cls(mutation=ypea.operators.NativeMutation(lambda x, rng: x, types=ypea.Integer))
    with pytest.raises(TypeError, match="crossover"):
        cls(crossover=ypea.operators.PolynomialMutation())
    with pytest.raises(ValueError, match="multi-objective"):
        cls(max_iter=2).solve(ypea.Problem(ypea.Real("x", shape=2), lambda s: float(s.x.sum())))
    with pytest.raises(ValueError, match="max_archive"):
        cls(max_archive=0)
    if cls is ypea.NSGAIII:
        for bad in (0, (3, 2, 1), "12", (3, 0)):
            with pytest.raises(ValueError, match="divisions"):
                cls(divisions=bad)


@pytest.mark.parametrize("cls", ALGS)
def test_other_operators(cls):
    alg = cls(max_nfe=2_000, crossover=ypea.operators.SBX(eta=5), mutation=ypea.operators.PolynomialMutation(prob=0.2))
    r = alg.solve(zdt1_problem(), seed=1)
    assert r.nfe == 2_000 and len(r.front) > 0


@pytest.mark.parametrize("cls", ALGS)
def test_maximization_mirrors_minimization(cls):
    a = cls(max_nfe=3_000).solve(zdt1_problem("min"), seed=4)
    b = cls(max_nfe=3_000).solve(zdt1_problem("max"), seed=4)
    assert np.array_equal(a.front.positions, b.front.positions)
    assert np.array_equal(a.front.obj_values, -b.front.obj_values)


def constrained_obj(sol):
    # ZDT1 with x1 >= 0.3: the front is the ZDT1 front cut at f1 = 0.3
    x = np.asarray(sol.x)
    return list(zdt1(x)), [0.3 - x[0]]


@pytest.mark.parametrize("cls", ALGS)
def test_constrained(cls):
    p = ypea.Problem(ypea.Real("x", shape=10), constrained_obj, n_obj=2)
    r = cls(max_nfe=5_000, pop_size=40).solve(p, seed=0)
    assert r.front.feasible.all()
    f = r.front.obj_values
    assert f[:, 0].min() >= 0.3 and f[:, 0].min() < 0.31
    assert np.all(f[:, 1] < 1 - np.sqrt(f[:, 0]) + 0.1)


@pytest.mark.parametrize("cls", ALGS)
def test_reproducible_and_picklable(cls):
    alg = cls(max_nfe=1_500, max_archive=30)
    a = alg.solve(zdt1_problem(), seed=11)
    b = pickle.loads(pickle.dumps(alg)).solve(zdt1_problem(), seed=11)
    assert len(a.front) <= 30
    assert np.array_equal(a.front.positions, b.front.positions)
    assert np.array_equal(a.history.trace_obj_value, b.history.trace_obj_value)


@pytest.mark.parametrize("cls", ALGS)
@pytest.mark.parametrize("budget", [1, 7, 123])
def test_budget(cls, budget):
    assert cls(max_nfe=budget).solve(zdt1_problem(), seed=0).nfe == budget


@pytest.mark.parametrize("cls", ALGS)
def test_every_variable_type(cls, mixed):
    p = ypea.Problem(mixed.vars, lambda s: [mixed_obj(s), float(np.sum(s.r))], n_obj=2)
    r = cls(max_iter=3, pop_size=20).solve(p, seed=0)
    for k in range(len(r.front)):
        sol = r.front[k].solution
        assert sorted(sol.p) == list(range(5)) and len(sol.q) == 3 and np.isclose(sol.fs.sum(), 1)
        assert np.allclose(r.front[k].obj_value, p.objective(sol))


@pytest.mark.parametrize("cls", ALGS)
def test_studio_generator(cls):
    alg = cls(max_nfe=800)
    a = alg.solve(zdt1_problem(), seed=ypea.StudioRng(42))
    b = alg.solve(zdt1_problem(), seed=ypea.StudioRng(42))
    assert a.nfe == 800 and np.array_equal(a.front.obj_values, b.front.obj_values)


@pytest.mark.parametrize("cls", ALGS)
def test_ask_tell_gives_the_same_front(cls):
    p = zdt1_problem()
    r = cls(max_nfe=600).solve(p, seed=2)
    run = cls(max_nfe=600).ask_tell(p, seed=2)
    while not run.done:
        run.tell(zdt1(run.ask().x))
    assert np.array_equal(run.result().front.obj_values, r.front.obj_values)
