"""Strength Pareto Evolutionary Algorithm 2 (SPEA2)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int
from ..operators import SBX, PolynomialMutation, check_space, native_variation
from ..population import Population


def distances(F):
    """Euclidean distances between the rows of ``F`` (n, m), infinite to itself and where
    a value is not finite."""
    F = np.asarray(F, dtype=float)
    with np.errstate(invalid="ignore"):
        d = np.sqrt(np.sum((F[:, None] - F[None]) ** 2, axis=-1))
    d[np.isnan(d)] = np.inf
    np.fill_diagonal(d, np.inf)
    return d


def spea2_fitness(dom, F, k=None):
    """SPEA2 fitness F(i) = R(i) + D(i) (smaller is better), Zitzler et al. (2001), 3.1.

    ``dom[i, j]`` is True when i dominates j, and ``F`` (n, m) holds the objective values.
    The strength S(i) is the number of solutions i dominates, the raw fitness R(i) the sum
    of the strengths of the solutions that dominate i, and the density
    D(i) = 1 / (sigma_k + 2), where sigma_k is the distance in objective space from i to
    its k-th nearest other solution. ``k`` defaults to floor(sqrt(n)).
    """
    dom = np.asarray(dom, dtype=bool)
    n = len(dom)
    strength = dom.sum(axis=1)
    raw = dom.T.astype(float) @ strength
    if n < 2:
        return raw + 0.0
    k = int(np.sqrt(n)) if k is None else k
    k = min(max(k, 1), n - 1)
    sigma = np.sort(distances(F), axis=1)[:, k - 1]
    return raw + 1 / (sigma + 2)


def truncate(F, size):
    """Indices of ``size`` rows of ``F`` kept by the SPEA2 truncation operator.

    One row at a time is removed: the row whose distance to its nearest neighbor is the
    smallest, ties broken by the second nearest, and so on (distances among the rows
    still kept). Of rows with equal lists, the first one is removed.
    """
    dist = distances(F)
    keep = np.arange(len(dist))
    while len(keep) > size:
        rows = np.sort(dist[np.ix_(keep, keep)], axis=1)
        keep = np.delete(keep, np.lexsort(rows.T[::-1])[0])
    return keep


class SPEA2(Algorithm):
    """Strength Pareto Evolutionary Algorithm 2 (SPEA2).

    SPEA2 keeps a population of ``pop_size`` solutions and an archive of ``archive_size``.
    Each iteration:

    1. Fitness: every solution of the population and the archive gets
       ``R(i) + D(i)``, smaller is better. The strength S(i) counts the solutions that i
       dominates, the raw fitness R(i) adds up the strengths of the solutions that dominate
       i (0 for a non-dominated one), and the density ``D(i) = 1 / (sigma_k + 2)`` uses the
       distance sigma_k in objective space to the k-th nearest other solution, with
       k = floor(sqrt(population plus archive size)). D is below 1/2, so a non-dominated
       solution has a fitness below 1.
    2. Environmental selection: the non-dominated solutions (fitness below 1) form the
       next archive. When they are too few, the best dominated ones by fitness fill it.
       When they are too many, the truncation operator removes, one at a time, the
       solution closest to another one (ties broken by the second closest distance, and
       so on), which keeps the boundary solutions.
    3. Mating: binary tournaments with replacement on the archive, by fitness, fill the
       mating pool.
    4. Variation: crossover and mutation of the pool give the next population.

    Parameters
    ----------
    pop_size : the population size N (default 100, as in the report's continuous tests)
    archive_size : the archive size N bar. None (the default) uses ``pop_size``, as in
        the report.
    crossover : ypea.operators.Crossover or None. The default (None) is simulated binary
        crossover with index 20, ``SBX()``, as in the report (its rate 0.9 is the NSGA-II
        setting, because the report gives none).
    mutation : ypea.operators.Mutation or None. The default (None) is polynomial mutation,
        ``PolynomialMutation()`` (index 20 and rate 1/n, the NSGA-II settings, because the
        report gives none).
    space : "encoded" (the default) applies ``crossover`` and ``mutation`` to the [0, 1]
        codes. "native" decodes each variable and applies its native operators, as GA does
        (permutation crossovers and moves, integer and selection operators, see the guide
        Native operators). Real variables keep SBX and polynomial mutation, so a problem
        with only Real variables runs exactly as with "encoded".
    operators : dict or None
        Native operators per variable name, as in GA (needs ``space="native"``).
    max_archive : int or None
        The largest size of the front (``Result.front``). None (the default) keeps every
        non-dominated solution found. This is not the algorithm's own archive, which is
        ``archive_size``.

    Dominance is the problem's own rule (constrained domination by default), so with
    constraints a feasible solution dominates an infeasible one. The front of the run
    (``Result.front``) is the exact archive of every non-dominated solution found, kept
    apart from the algorithm's archive.

    Reference: E. Zitzler, M. Laumanns and L. Thiele (2001), SPEA2: Improving the strength
    Pareto evolutionary algorithm, TIK-Report 103, ETH Zurich.
    """

    name = "Strength Pareto Evolutionary Algorithm 2"
    short_name = "SPEA2"
    objectives = "multi"

    def __init__(self, *, archive_size=None, crossover=None, mutation=None, space="encoded", operators=None,
                 max_archive=None, **kwargs):
        self.archive_size = archive_size
        self.crossover = crossover
        self.mutation = mutation
        self.space = space
        self.operators = operators
        self.max_archive = max_archive
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(self.archive_size is None or is_int(self.archive_size, 1),
              "archive_size must be a positive integer or None")
        check_space(self)

    def check_problem(self, problem):
        check_space(self, problem)

    def initialize(self):
        if self.space == "native":
            self._crossover, self._mutation = native_variation(self.problem, self.operators, self.crossover,
                                                               self.mutation, SBX(), PolynomialMutation())
        else:
            self._crossover = self.crossover or SBX()
            self._mutation = self.mutation or PolynomialMutation()
        self.pop = self.random_population()
        self.archive, self.fitness = self._select(self.pop)

    def _select(self, union):
        c = self.problem.comparator
        F, V = union.F, union.V
        dom = c.dominates(F[:, None], V[:, None], F[None], V[None])
        Fm = c.sign * F
        fit = spea2_fitness(dom, Fm)
        size = self.archive_size or self.pop_size
        nondom = np.flatnonzero(fit < 1)
        if len(nondom) > size:
            chosen = nondom[truncate(Fm[nondom], size)]
        else:
            chosen = np.argsort(fit, kind="stable")[:size]
        return union[chosen], fit[chosen]

    def iterate(self):
        n, a = self.pop_size, len(self.archive)
        pairs = (n + 1) // 2
        i = self.rng.integers(a, size=(2, 2 * pairs))
        winner = np.where(self.fitness[i[0]] <= self.fitness[i[1]], i[0], i[1])
        X = self.archive.X[winner]
        c1, c2 = self._crossover(X[:pairs], X[pairs:], self.rng)
        children = self._mutation(np.concatenate([c1, c2])[:n], self.rng)
        self.pop = self.evaluate(children)
        self.archive, self.fitness = self._select(Population.concat(self.archive, self.pop))
