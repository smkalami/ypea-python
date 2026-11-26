"""Non-dominated Sorting Genetic Algorithms: NSGA-II and NSGA-III."""

from __future__ import annotations

from math import comb

import numpy as np

from ..algorithm import Algorithm, check, is_int
from ..benchmarks.multi import das_dennis
from ..operators import SBX, Operator, PolynomialMutation, check_operator
from ..population import Population


def crowded_tournament(rng, rank, crowd, n):
    """Indices of the winners of ``n`` binary tournaments by the crowded comparison of
    Deb et al. (2002): the lower front wins, then the larger crowding distance. A full tie
    is decided by a fair coin. The competitors are drawn with replacement."""
    a, b = rng.integers(len(rank), size=(2, n))
    coin = rng.random(n) < 0.5
    tie = (rank[a] == rank[b]) & (crowd[a] == crowd[b])
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b])) | (tie & coin)
    return np.where(a_wins, a, b)


# the paper's Table I for 3, 5, 8, 10 and 15 objectives, our choice for the others
def default_divisions(m):
    """Default divisions of the NSGA-III reference points for ``m`` objectives: one number
    for one layer, or (boundary, inside) for two layers."""
    if m <= 5:
        return {2: 99, 3: 12, 4: 8, 5: 6}[m]
    return (3, 2) if m <= 10 else (2, 1)


def reference_points(m, divisions=None):
    """Structured reference points of NSGA-III on the unit simplex (Das and Dennis 1998).

    ``divisions`` is p, the number of divisions of each objective (comb(m + p - 1, p)
    points), or a pair (p_boundary, p_inside) for two layers: the inside layer is shrunk
    by half toward the center of the simplex (Deb and Jain 2014, Fig. 4). None gives
    ``default_divisions(m)``.
    """
    p = default_divisions(m) if divisions is None else divisions
    layers = [das_dennis(comb(q + m - 1, m - 1), m) for q in ((p,) if is_int(p) else p)]
    if len(layers) == 2:
        layers[1] = 0.5 * layers[1] + 0.5 / m
    return np.vstack(layers)


def normalize(f, ideal=None, extremes=None, first=None):
    """Adaptive normalization of NSGA-III (Deb and Jain 2014, Algorithm 2).

    ``f`` (n, m) holds the objective values of S_t, all to be minimized. ``ideal`` and
    ``extremes`` (m, m) are the ideal point and the extreme points found so far (None at
    the start), and ``first`` marks the members of the first front. Rows with non-finite
    values are left out of every step and come back as NaN.

    The ideal point is the smallest value of each objective so far. The extreme point of
    axis j minimizes ASF(x, w^j) = max_i f'_i(x) / w^j_i over the previous extreme points and
    S_t, with w^j_i = 1e-6 except w^j_j = 1. The intercepts come from the hyperplane through
    the extreme points. When it does not exist (singular), or an intercept is not finite or
    at most 1e-6, each intercept is the largest translated value in the first front, and an
    intercept still at most 1e-6 is the largest translated value in S_t (the paper does not
    say what to do then: this is the rule of pymoo).

    Returns (fn, ideal, extremes, intercepts) with fn = (f - ideal) / intercepts.
    """
    f = np.asarray(f, dtype=float)
    n, m = f.shape
    fin = np.isfinite(f).all(axis=1)
    fn = np.full((n, m), np.nan)
    if not fin.any():
        return fn, ideal, extremes, None
    ff = f[fin]
    ideal = ff.min(axis=0) if ideal is None else np.minimum(ideal, ff.min(axis=0))
    cand = ff if extremes is None else np.vstack([extremes, ff])
    w = np.full((m, m), 1e-6)
    np.fill_diagonal(w, 1.0)
    asf = np.max((cand - ideal)[None, :, :] / w[:, None, :], axis=2)  # (axis, candidate)
    extremes = cand[np.argmin(asf, axis=1)]
    a = _intercepts(extremes - ideal)
    if a is None:
        worst = lambda mask: (ff[mask] - ideal).max(axis=0)  # noqa: E731
        a = worst(np.ones(len(ff), bool) if first is None or not first[fin].any() else first[fin])
        a = np.where(a > 1e-6, a, worst(np.ones(len(ff), bool)))
    a = np.maximum(a, 1e-12)  # an objective with no range at all
    fn[fin] = (ff - ideal) / a
    return fn, ideal, extremes, a


def _intercepts(e):
    """Intercepts of the hyperplane through the translated extreme points (rows of e), or None."""
    one = np.ones(len(e))
    try:
        b = np.linalg.solve(e, one)
    except np.linalg.LinAlgError:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        a = 1 / b
    if not (np.allclose(e @ b, one) and np.all(np.isfinite(a)) and np.all(a > 1e-6)):
        return None
    return a


def associate(fn, ref):
    """Association of NSGA-III (Algorithm 3): for each normalized point, the reference line
    (through the origin and a reference point) at the smallest perpendicular distance, and
    that distance. Rows with NaN get line 0 and an infinite distance."""
    u = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    proj = fn @ u.T
    d = np.linalg.norm(fn[:, None, :] - proj[:, :, None] * u[None], axis=2)
    bad = np.isnan(d).any(axis=1)
    d[bad] = np.inf
    pi = np.argmin(d, axis=1)
    return pi, d[np.arange(len(fn)), pi]


def niching(k, pi, dist, last, rho, rng):
    """Niche preserving selection of NSGA-III (Algorithm 4).

    Picks ``k`` members of the last front (``last`` is its mask) one at a time. Each step
    takes a reference point with the smallest niche count ``rho`` (a random one of ties).
    If no member of the last front is associated with it, the point is dropped for this
    generation. Otherwise, when its count is 0 the member closest to its line is taken,
    and else a random one. Returns the indices of the picked members, in order.
    """
    rho = np.array(rho, dtype=int)
    active = np.ones(len(rho), bool)
    free = np.array(last, dtype=bool)
    out = []
    while len(out) < k:
        jmin = np.flatnonzero(active & (rho == rho[active].min()))
        j = jmin[rng.integers(len(jmin))]
        members = np.flatnonzero(free & (pi == j))
        if not len(members):
            active[j] = False
            continue
        s = members[np.argmin(dist[members])] if rho[j] == 0 else members[rng.integers(len(members))]
        out.append(s)
        free[s] = False
        rho[j] += 1
    return np.array(out, dtype=int)


class NSGAII(Algorithm):
    """Non-dominated Sorting Genetic Algorithm II for several objectives.

    Each iteration makes ``pop_size`` children: parents are chosen by binary tournaments
    on the crowded comparison (the lower front wins, then the larger crowding distance),
    each pair is crossed, and every child is mutated. Parents and children are merged and
    sorted into non-dominated fronts. The best fronts fill the next population, and the
    last front that does not fit whole is cut by crowding distance, the largest first.
    The front and crowding distance of each survivor are those computed in the merged
    population, as in the paper.

    Constraints use constrained domination (Deb et al. 2002, Section VI) through the
    problem's comparator: a feasible solution dominates an infeasible one, and of two
    infeasible ones the smaller violation dominates. The same order drives the sort and the
    tournaments, so the constrained tournament of the paper comes for free.

    Parameters
    ----------
    pop_size : population size N (default 100). Each iteration makes N children.
    crossover : ypea.operators.Crossover or None
        A crossover on the [0, 1] codes, called as ``op(x1, x2, rng)``. None (the default)
        is simulated binary crossover with the paper's settings, ``SBX(eta=20, prob=0.9)``.
    mutation : ypea.operators.Mutation or None
        A mutation on the [0, 1] codes, called as ``op(x, rng)``. None (the default) is
        polynomial mutation with the paper's settings, ``PolynomialMutation(eta=20)``, which
        changes each of the n codes with probability 1/n.
    max_archive : int or None
        The largest size of the front kept in the result (see ``RandomSearch``). None (the
        default) keeps every non-dominated solution found.

    The operators work on the codes, so every variable type can be used. Native operators
    (``NativeCrossover``, ``NativeMutation``) are not supported.

    Differences from the paper: the initial population also gets crowding distances, so
    the first tournaments use them too (the paper ranks it by front only). A tournament
    between two solutions with the same front and distance is decided by a coin.

    Reference: K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
    multiobjective genetic algorithm: NSGA-II", IEEE Transactions on Evolutionary
    Computation 6(2), 182 to 197, 2002.
    """

    name = "Non-dominated Sorting Genetic Algorithm II"
    short_name = "NSGA-II"
    objectives = "multi"

    def __init__(self, *, crossover=None, mutation=None, max_archive=None, **kwargs):
        self.crossover = crossover
        self.mutation = mutation
        self.max_archive = max_archive
        super().__init__(**kwargs)

    def default_operators(self):
        """The paper's operators, used when ``crossover`` or ``mutation`` is None."""
        return SBX(eta=20.0, prob=0.9), PolynomialMutation(eta=20.0)

    def validate(self):
        super().validate()
        for op, kind in ((self.crossover, "crossover"), (self.mutation, "mutation")):
            check(not (isinstance(op, Operator) and op.space != "code"),
                  f"{self.short_name} works on the [0, 1] codes, so {kind} must be a code operator "
                  "such as SBX() or PolynomialMutation()")
            check_operator(op, kind, "encoded")

    def check_problem(self, problem):
        check_operator(self.crossover, "crossover", "encoded", problem)
        check_operator(self.mutation, "mutation", "encoded", problem)

    def initialize(self):
        cx, mut = self.default_operators()
        self._cx = self.crossover or cx
        self._mut = self.mutation or mut
        self._n = self.population_size()
        self.pop = self.random_population(self._n)
        self._rank, self._crowd = self._rank_and_crowd(self.pop)

    def iterate(self):
        n = self._n
        idx = self.select_parents(2 * ((n + 1) // 2))
        y1, y2 = self._cx(self.pop.X[idx[0::2]], self.pop.X[idx[1::2]], self.rng)
        y = np.empty((2 * len(y1), self.dim))
        y[0::2], y[1::2] = y1, y2
        off = self.evaluate(self._mut(y[:n], self.rng))
        self.survive(Population.concat(self.pop, off))

    def population_size(self):
        """The population size of the run."""
        return self.pop_size

    def select_parents(self, k):
        """Indices of ``k`` parents (pairs are taken in order)."""
        return crowded_tournament(self.rng, self._rank, self._crowd, k)

    def _rank_and_crowd(self, pop):
        rank = self.nondominated_sort(pop)
        with np.errstate(invalid="ignore"):  # rows past max_nfe have infinite values
            return rank, self.crowding_distance(pop, rank)

    def survive(self, merged):
        """Elitist selection of the next population from parents and children."""
        rank, crowd = self._rank_and_crowd(merged)
        keep = np.lexsort((-crowd, rank))[:self._n]  # by front, then the largest distance first
        self.pop, self._rank, self._crowd = merged[keep], rank[keep], crowd[keep]


class NSGAIII(NSGAII):
    """Reference point based Non-dominated Sorting Genetic Algorithm III for many objectives.

    It is NSGA-II with a different way to cut the last front and no tournament. Each
    iteration makes ``pop_size`` children from parents picked at random, crossed and
    mutated. Parents and children are merged and sorted into non-dominated fronts, and
    the best fronts fill S_t until it holds at least N solutions. When S_t has more, the
    members of S_t are normalized, each is associated with its nearest reference line, and
    the free places go one at a time to the least crowded reference points (niching).

    Normalization (Algorithm 2 of the paper): the objectives are translated by the ideal
    point (the smallest value of each objective found in every S_t so far), the extreme
    point of each axis minimizes the achievement scalarizing function among S_t and the
    extreme points found before, and the intercepts of the hyperplane through the extreme
    points scale each objective. See ``normalize`` for what happens when the hyperplane
    does not exist.

    Parameters
    ----------
    divisions : int, pair of ints, or None
        The number of divisions p of each objective for the reference points of Das and
        Dennis: comb(m + p - 1, p) points. A pair (p_boundary, p_inside) gives two layers,
        the inside one shrunk by half toward the center, as the paper does for many
        objectives. None (the default) gives the paper's choice (Table I): 12 for 3
        objectives (91 points), 6 for 5 (210), (3, 2) for 8 and 10 (156 and 275), (2, 1)
        for 15 (135). For the other numbers of objectives: 99 for 2 (100 points), 8 for 4
        (165), (3, 2) for 6 to 10, and (2, 1) above 10.
    pop_size : population size N. None (the default) gives the smallest multiple of 4 at
        least as large as the number of reference points, as in Table I of the paper.
    crossover : ypea.operators.Crossover or None
        None (the default) is ``SBX(eta=30, prob=1.0)``, the setting of Table II.
    mutation : ypea.operators.Mutation or None
        None (the default) is ``PolynomialMutation(eta=20)`` with probability 1/n.
    max_archive : int or None
        The largest size of the front kept in the result. None keeps all of it.

    Constraints: the fronts come from the problem's comparator, so with constraints they
    follow constrained domination (feasible solutions first, then the smaller violation).
    This is not the constrained NSGA-III of part II of the paper, which changes the
    selection and the niching for infeasible solutions.

    Differences from the paper: the paper's Table I has no setting for 2, 4, 6, 7, 9 or
    more than 10 (other than 15) objectives, so those defaults are our choice. When the
    hyperplane through the extreme points does not exist, the paper does not say what to
    do, and the intercepts fall back to the largest values in the first front (see
    ``normalize``). The paper's text (Section IV C) keeps the ideal and extreme points over
    the whole run, while Algorithm 2 computes them from S_t alone. This class follows the
    text, so it updates both in every iteration. Aspiration points supplied by the user
    are not supported.

    Reference: K. Deb and H. Jain, "An evolutionary many-objective optimization algorithm
    using reference-point-based nondominated sorting approach, part I: solving problems
    with box constraints", IEEE Transactions on Evolutionary Computation 18(4), 577 to
    601, 2014.
    """

    name = "Non-dominated Sorting Genetic Algorithm III"
    short_name = "NSGA-III"
    auto_pop_size = True

    def __init__(self, *, divisions=None, pop_size=None, **kwargs):
        self.divisions = divisions
        super().__init__(pop_size=pop_size, **kwargs)

    def default_operators(self):
        return SBX(eta=30.0, prob=1.0), PolynomialMutation(eta=20.0)

    def validate(self):
        super().validate()
        d = self.divisions
        layers = (d,) if is_int(d) else tuple(d) if isinstance(d, (tuple, list)) else None
        check(d is None or (layers and len(layers) <= 2 and all(is_int(p, 1) for p in layers)),
              "divisions must be a positive integer, a pair of them, or None")

    def initialize(self):
        self._ref = reference_points(self.problem.n_obj, self.divisions)
        self._ideal = self._extremes = None
        super().initialize()

    def population_size(self):
        return -(-len(self._ref) // 4) * 4 if self.pop_size is None else self.pop_size

    @property
    def ref_points(self):
        """The reference points of the current run, shape (H, m)."""
        return self._ref

    def select_parents(self, k):
        return self.rng.integers(len(self.pop), size=k)

    def survive(self, merged):
        n = self._n
        rank = self.nondominated_sort(merged)
        last = np.sort(rank)[n - 1]
        s = np.flatnonzero(rank <= last)
        f = self.problem.comparator.sign * merged.F[s]  # every objective to be minimized
        fn, self._ideal, self._extremes, _ = normalize(f, self._ideal, self._extremes, rank[s] == 0)
        if len(s) > n:
            pi, dist = associate(fn, self._ref)
            before = rank[s] < last
            rho = np.bincount(pi[before], minlength=len(self._ref))
            picked = niching(n - before.sum(), pi, dist, ~before, rho, self.rng)
            s = np.r_[s[before], s[picked]]
        self.pop = merged[s]
