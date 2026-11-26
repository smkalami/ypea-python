"""Multiobjective Evolutionary Algorithm Based on Decomposition (MOEA/D)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real
from ..benchmarks.multi import das_dennis
from ..comparison import Comparator
from ..operators import SBX, PolynomialMutation, check_operator

DECOMPOSITIONS = ("tchebycheff", "weighted_sum", "pbi")


def aggregate(F, weights, ideal, method="tchebycheff", theta=5.0):
    """Value of each subproblem (smaller is better), Zhang and Li (2007), Section II.

    ``F`` (n, m) holds objective values turned into minimization, ``weights`` (n, m) or
    (m,) the weight vectors and ``ideal`` (m,) the reference point z.

    * "tchebycheff": max_i w_i |f_i - z_i|
    * "weighted_sum": sum_i w_i f_i
    * "pbi": d1 + theta d2, where d1 = (f - z) . w / |w| is the distance along w and d2
      the distance from f to the line through z with direction w

    A zero weight times an infinite (worst) value counts as 0, and NaN as infinity.
    """
    F, w = np.asarray(F, dtype=float), np.asarray(weights, dtype=float)
    diff = F - np.asarray(ideal, dtype=float)
    with np.errstate(invalid="ignore"):
        if method == "tchebycheff":
            g = np.max(np.where(w > 0, w * np.abs(diff), 0.0), axis=-1)
        elif method == "weighted_sum":
            g = np.sum(np.where(w > 0, w * F, 0.0), axis=-1)
        else:
            u = w / np.linalg.norm(w, axis=-1, keepdims=True)
            d1 = np.sum(diff * u, axis=-1)
            d2 = np.linalg.norm(diff - d1[..., None] * u, axis=-1)
            g = d1 + theta * d2
    return np.where(np.isnan(g), np.inf, g)


class MOEAD(Algorithm):
    """Multiobjective Evolutionary Algorithm Based on Decomposition (MOEA/D).

    MOEA/D turns the problem into N scalar subproblems, one per weight vector, and keeps
    one solution for each. The weight vectors are the Das and Dennis lattice on the unit
    simplex, and each subproblem has a neighborhood: the T subproblems with the closest
    weight vectors (itself included). Each iteration visits the subproblems in order. For
    subproblem i it picks two different neighbors at random, crosses their solutions to
    one child and mutates it, evaluates the child, updates the ideal point z (the best value
    of each objective found so far), and then the child replaces the solution of every
    neighbor j whose subproblem it solves at least as well.

    Parameters
    ----------
    pop_size : the number of subproblems wanted (default 100). The weights are the
        largest Das and Dennis lattice with at most ``pop_size`` points, so the population
        may be smaller: for 3 objectives, 100 gives 91 subproblems and 300 gives 300. For
        2 objectives every size is exact.
    neighbors : T, the size of each neighborhood (default 20, as in the paper). It is
        reduced to the number of subproblems when larger.
    decomposition : "tchebycheff" (the default, used in the paper's continuous tests),
        "weighted_sum" (reaches only the convex parts of a front) or "pbi" (penalty based
        boundary intersection).
    theta : the penalty factor of "pbi" (default 5, as in the paper).
    crossover : ypea.operators.Crossover or None. The default (None) is the paper's
        simulated binary crossover with index 20 and rate 1, ``SBX(prob=1.0)``. Of the two
        children of SBX the first is kept (SBX swaps them at random).
    mutation : ypea.operators.Mutation or None. The default (None) is the paper's
        polynomial mutation with index 20 and rate 1/n, ``PolynomialMutation()``.
    batch : bool
        False (the default) follows the paper: each child is evaluated on its own and
        updates the population before the next child is bred. True is a variant for costly
        or parallel objectives: each iteration breeds all N children from the population
        as it stands at the start of the iteration, evaluates them in one call (so a
        vectorized objective or an executor gets N solutions at once), and then applies the
        ideal point and neighbor updates of the children in subproblem order. The variant
        is not in the paper. Its children cannot use the improvements made earlier in the
        same iteration, so it may converge a little slower per evaluation.
    max_archive : int or None
        The largest size of the front (``Result.front``). None (the default) keeps every
        non-dominated solution found.

    The paper's external population EP is not kept, because every run already keeps the
    exact archive of non-dominated solutions (``Result.front``). The objectives are used
    as they are (no normalization), with each objective turned into minimization first.

    As in the paper, each child is evaluated on its own by default, so the objective is
    called once per evaluation, and a vectorized objective or an executor gives no speed
    up unless ``batch=True``.

    **Constraints.** A child is compared with a neighbor's solution by the problem's own
    rule, applied to the subproblem values: with Deb's feasibility rules the smaller total
    violation wins and equal violations compare the subproblem values, and with
    ``constraints="penalty"`` the values ``g + penalty * violation`` are compared. The child
    replaces the solution when it is at least as good (the paper's "less than or equal").
    The ideal point uses every evaluated solution, feasible or not.

    Reference: Q. Zhang and H. Li (2007), MOEA/D: A multiobjective evolutionary algorithm
    based on decomposition, IEEE Transactions on Evolutionary Computation 11(6), 712 to 731.
    I. Das and J. E. Dennis (1998), Normal-boundary intersection, SIAM Journal on
    Optimization 8(3), for the weight vectors.
    """

    name = "Multiobjective Evolutionary Algorithm Based on Decomposition"
    short_name = "MOEA/D"
    objectives = "multi"

    def __init__(self, *, neighbors=20, decomposition="tchebycheff", theta=5.0, crossover=None, mutation=None,
                 batch=False, max_archive=None, **kwargs):
        self.neighbors = neighbors
        self.decomposition = decomposition
        self.theta = theta
        self.crossover = crossover
        self.mutation = mutation
        self.batch = batch
        self.max_archive = max_archive
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_int(self.neighbors, 2), "neighbors must be an integer of at least 2")
        check(self.decomposition in DECOMPOSITIONS, f"decomposition must be one of {DECOMPOSITIONS}")
        check(is_real(self.theta, 0), "theta must be non negative")
        check(isinstance(self.batch, bool), "batch must be True or False")
        check_operator(self.crossover, "crossover", "encoded")
        check_operator(self.mutation, "mutation", "encoded")

    def check_problem(self, problem):
        m = problem.n_obj
        check(self.pop_size >= m, f"MOEA/D needs a pop_size of at least the number of objectives ({m})")

    def initialize(self):
        c = self.problem.comparator
        self._cmp = Comparator("min", c.constraint_handling, c.penalty)
        self.weights = das_dennis(self.pop_size, self.problem.n_obj)
        n = len(self.weights)
        t = min(self.neighbors, n)
        dist = np.linalg.norm(self.weights[:, None] - self.weights[None], axis=-1)
        self.neighborhood = np.argsort(dist, axis=1, kind="stable")[:, :t]
        self._crossover = self.crossover or SBX(prob=1.0)
        self._mutation = self.mutation or PolynomialMutation()
        self.pop = self.random_population(n)
        self.ideal = np.min(self._minimized(self.pop.F), axis=0)

    def _minimized(self, F):
        f = self.problem.comparator.sign * F
        return np.where(np.isnan(f), np.inf, f)

    def _g(self, F, w):
        return aggregate(self._minimized(F), w, self.ideal, self.decomposition, self.theta)

    def iterate(self):
        pop, B = self.pop, self.neighborhood
        n, t = B.shape
        # two different neighbors per subproblem, drawn for the whole iteration
        k = self.rng.integers(t, size=n)
        k2 = (k + 1 + self.rng.integers(t - 1, size=n)) % t
        if self.batch:
            rows = np.arange(n)
            y = self._crossover(pop.X[B[rows, k]], pop.X[B[rows, k2]], self.rng)[0]
            start = self.nfe
            children = self.evaluate(self._mutation(y, self.rng))
            for i in range(self.nfe - start):  # rows past max_nfe were not evaluated
                self._update(i, children[i])
            return
        for i in range(n):
            a, b = B[i, k[i]], B[i, k2[i]]
            y = self._crossover(pop.X[a:a + 1], pop.X[b:b + 1], self.rng)[0]
            self._update(i, self.evaluate(self._mutation(y, self.rng)))
            if self.evaluator.stop_reason:
                break

    def _update(self, i, child):
        """The ideal point, then the neighbors of subproblem i that the child solves at least as well."""
        pop = self.pop
        self.ideal = np.minimum(self.ideal, self._minimized(child.F[0]))
        js = self.neighborhood[i]
        w = self.weights[js]
        g_new, g_old = self._g(np.repeat(child.F, len(js), 0), w), self._g(pop.F[js], w)
        # replace where the child is no worse: not (old better than child)
        take = js[~self._cmp.better(g_old[:, None], pop.V[js], g_new[:, None], np.repeat(child.V, len(js)))]
        pop.X[take], pop.F[take], pop.V[take] = child.X[0], child.F[0], child.V[0]
