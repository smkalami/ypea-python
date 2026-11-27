"""Multi-Objective Particle Swarm Optimization (MOPSO)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real
from ..population import Population


def hypercubes(F, divisions):
    """The hypercube of each row of ``F`` (n, m) in an adaptive grid, and the counts.

    Each objective's range in ``F`` is split into ``divisions`` equal parts. Returns the
    index of each row's hypercube among the occupied ones, and the number of rows in each
    occupied hypercube. Non-finite values go to the last part, and an objective with no
    range puts every row in its first part.
    """
    F = np.asarray(F, dtype=float)
    fin = np.isfinite(F)
    lo = np.min(np.where(fin, F, np.inf), axis=0)
    hi = np.max(np.where(fin, F, -np.inf), axis=0)
    span = hi - lo
    ok = np.isfinite(span) & (span > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cell = np.floor((F - lo) / np.where(ok, span, 1.0) * divisions)
    cell = np.where(fin, np.where(ok, cell, 0), divisions - 1)
    cell = np.clip(cell, 0, divisions - 1).astype(np.int64)
    _, index, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    return index.reshape(-1), counts


def leader_probabilities(counts):
    """Roulette probabilities of the occupied hypercubes: fitness 10 / count, normalized.
    (The paper's x = 10 cancels out, so any x > 1 gives the same probabilities.)"""
    fit = 10.0 / np.asarray(counts, dtype=float)
    return fit / fit.sum()


def select_leaders(index, counts, n, rng):
    """Rows of ``n`` leaders in a repository whose members lie in the hypercubes ``index``
    (from ``hypercubes``): a hypercube by roulette (``leader_probabilities``), then one of
    its members uniformly at random."""
    cube = rng.choice(len(counts), size=n, p=leader_probabilities(counts))
    order = np.argsort(index, kind="stable")  # members grouped by hypercube
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return order[start[cube] + rng.integers(counts[cube], size=n)]


def mutation_probability(t, rate):
    """The mutation probability and range ``(1 - t) ** (5 / rate)`` when a share t of the
    run is used (Fig. 5 of the paper), and 0 when ``rate`` is 0."""
    return (1 - t) ** (5 / rate) if rate > 0 else 0.0


def mutate(X, p, rng):
    """The mutation of Fig. 5 of the paper on codes in [0, 1], in place: with probability
    ``p`` a row gets one random component redrawn uniformly within ``p`` of its value
    (clipped to [0, 1]). Returns ``X``."""
    n, d = X.shape
    flip = rng.random(n) < p
    dim = rng.integers(d, size=n)
    u = rng.random(n)
    rows = np.flatnonzero(flip)
    x = X[rows, dim[rows]]
    lo, hi = np.maximum(x - p, 0.0), np.minimum(x + p, 1.0)
    X[rows, dim[rows]] = lo + u[rows] * (hi - lo)
    return X


class MOPSO(Algorithm):
    """Multi-Objective Particle Swarm Optimization (MOPSO) of Coello Coello, Pulido and
    Lechuga (2004).

    The swarm keeps a repository of the non-dominated positions found, at most
    ``repository_size`` of them, located in an adaptive grid of hypercubes in objective
    space. Each iteration:

    1. Each particle takes a leader from the repository: a hypercube is chosen by roulette,
       with fitness 10 divided by the number of members in it (so sparse regions are
       preferred), and then a member of that hypercube at random.
    2. ``v = w v + r1 (pbest - x) + r2 (leader - x)`` and ``x = x + v``, with r1 and r2
       uniform in [0, 1] for each component. A component that leaves [0, 1] is set to the
       bound and its velocity is reversed.
    3. Mutation: with probability ``p = (1 - t) ** (5 / mutation_rate)``, where t is the
       share of the run used, one random component of a particle is redrawn uniformly
       within ``p`` of its value (clipped to [0, 1]). So every particle mutates over the
       full range at the start, and mutation fades out over the run.
    4. The particles are evaluated and the repository is updated: the non-dominated
       positions of the repository and the swarm are kept. When there are more than
       ``repository_size``, a random member of the most crowded hypercube is removed, and
       the grid is computed again, until they fit.
    5. A particle's best position is replaced when the new position dominates it, kept
       when it dominates the new one, and otherwise replaced with probability 1/2.

    Parameters
    ----------
    pop_size : number of particles (default 100, as in the paper)
    w : inertia weight (default 0.7). The paper uses 0.4. On 21 ZDT, DTLZ and WFG problems
        (15 runs, 25 000 evaluations), 0.7 gave a better IGD on 13 and a worse one on 6.
    repository_size : the largest size of the leader repository (default 100)
    divisions : number of grid divisions per objective (default 30)
    mutation_rate : sets how fast mutation fades, through the exponent 5 / mutation_rate
        (default 0.5). 0 turns mutation off.
    max_archive : int or None
        The largest size of the front (``Result.front``). None (the default) keeps every
        non-dominated solution found.

    The repository guides the swarm only. The front of the run (``Result.front``) is the
    exact archive of every non-dominated solution found, kept apart from it.

    The paper leaves some details open. Here:

    - The grid spans the range of the repository in each objective, each objective turned
      into minimization, and is computed again after every change (the paper moves the
      grid bounds only when a new member falls outside them).
    - Hypercubes with one member get fitness 10 like the others (the paper names those
      with more than one member).
    - r1 and r2 are drawn for each component, and mutation acts after the move and before
      the evaluation.
    - A full repository drops a random member of its most crowded hypercube, and of
      members with equal objective values and violation only the first is kept.

    Comparisons use the problem's dominance rule. With the default constraint handling
    this is the paper's rule: a feasible solution beats an infeasible one, and of two
    infeasible ones the smaller violation wins.

    Reference: C. A. Coello Coello, G. T. Pulido and M. S. Lechuga (2004), Handling
    multiple objectives with particle swarm optimization, IEEE Transactions on
    Evolutionary Computation 8(3), 256 to 279.
    """

    name = "Multi-Objective Particle Swarm Optimization"
    short_name = "MOPSO"
    objectives = "multi"

    def __init__(self, *, w=0.7, repository_size=100, divisions=30, mutation_rate=0.5,
                 max_archive=None, **kwargs):
        self.w = w
        self.repository_size = repository_size
        self.divisions = divisions
        self.mutation_rate = mutation_rate
        self.max_archive = max_archive
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_real(self.w, 0), "w must be non negative")
        check(is_int(self.repository_size, 1), "repository_size must be a positive integer")
        check(is_int(self.divisions, 1), "divisions must be a positive integer")
        check(is_real(self.mutation_rate, 0), "mutation_rate must be non negative")

    def initialize(self):
        self.pop = self.random_population()
        self.velocity = np.zeros_like(self.pop.X)
        self.pbest = self.pop.copy()
        self.repository = None
        self.repository = self._update_repository(self.pop)

    def _cubes(self, rep):
        return hypercubes(self.problem.comparator.sign * rep.F, self.divisions)

    def _update_repository(self, new):
        rep = new if self.repository is None else Population.concat(self.repository, new)
        rep = rep[np.flatnonzero(self.nondominated_sort(rep) == 0)]
        # equal objective values and violation: keep the first copy only
        _, first = np.unique(np.column_stack([rep.F, rep.V]), axis=0, return_index=True)
        rep = rep[np.sort(first)]
        while len(rep) > self.repository_size:
            index, counts = self._cubes(rep)
            members = np.flatnonzero(index == np.argmax(counts))
            drop = members[self.rng.integers(len(members), size=1)[0]]
            rep = rep[np.flatnonzero(np.arange(len(rep)) != drop)]
        return rep

    def iterate(self):
        X, V, rep = self.pop.X, self.velocity, self.repository
        n, d = X.shape
        index, counts = self._cubes(rep)
        leader = rep.X[select_leaders(index, counts, n, self.rng)]
        r1, r2 = self.rng.random((2, n, d))
        V = self.w * V + r1 * (self.pbest.X - X) + r2 * (leader - X)
        X = X + V
        out = (X < 0) | (X > 1)
        V[out] = -V[out]
        X = np.clip(X, 0.0, 1.0)
        X = mutate(X, mutation_probability(self.progress(), self.mutation_rate), self.rng)
        self.velocity = V
        self.pop = self.evaluate(X)
        self.repository = self._update_repository(self.pop)
        better, worse = self.dominates(self.pop, self.pbest), self.dominates(self.pbest, self.pop)
        coin = self.rng.random(n) < 0.5
        self.pbest.replace(better | (~worse & coin), self.pop)
