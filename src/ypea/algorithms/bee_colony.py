"""Artificial Bee Colony (ABC)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real

FITNESS_MODES = ("auto", "paper", "rank")


class ABC(Algorithm):
    """Artificial Bee Colony (Karaboga, 2005, and Karaboga and Basturk, 2007).

    Each food source (a member of the population of ``pop_size``) has one employed bee.
    A bee changes one random component j of its source: v_j = x_j + phi (x_j - x_kj),
    where k is another random source and phi is uniform in
    [-max_acceleration, max_acceleration]. The new position replaces the source only if
    it is better (greedy selection). Onlooker bees then pick sources with probability
    proportional to their fitness and search around them the same way. A source that
    fails to improve ``trial_limit`` times in a row is abandoned, and a scout bee replaces
    it with a random solution.

    Parameters
    ----------
    pop_size : number of food sources (the paper's SN)
    onlooker_count : number of onlooker bees per iteration. None (the default) means SN
        (pop_size), as in the paper. YPEA for MATLAB used 100.
    max_acceleration : the limit a of the uniform factor phi in [-a, a]
    trial_limit : abandonment limit. None (the default) means onlooker_count * d, where d is
        the code size, as in Karaboga (2005, Table 2). YPEA for MATLAB used
        round(0.6 * d * pop_size).
    fitness : "auto" (default), "paper" or "rank". The paper uses 1 / (1 + f) for
        f >= 0 and 1 + |f| for f < 0 (on the objective turned into "smaller is better").
        "auto" uses that formula, but switches to ranks when the problem has constraints,
        because an infeasible solution has no meaningful objective value. With ranks the
        selection probabilities are ``selection_probs`` of the population.

    Differences from the paper
    --------------------------
    The bees of each phase work synchronously, so a phase is evaluated as one batch.
    Every employed bee uses the sources as they were at the start of the phase. All
    onlookers choose their sources at once, and each source keeps the best of the
    candidates its onlookers found if that is better (greedy selection). Its trial
    counter then resets to 0, or else grows by the number of onlookers that visited it.
    At most one scout appears per iteration: the source with the largest
    trial counter, if that counter has reached the limit.

    References: Karaboga, D. (2005). An idea based on honey bee swarm for numerical
    optimization. Technical report TR06, Erciyes University. Karaboga, D. and Basturk, B.
    (2007). A powerful and efficient algorithm for numerical function optimization:
    artificial bee colony (ABC) algorithm. Journal of Global Optimization, 39, 459 to 471.
    """

    name = "Artificial Bee Colony"
    short_name = "ABC"

    def __init__(self, *, onlooker_count=None, max_acceleration=1.0, trial_limit=None,
                 fitness="auto", **kwargs):
        self.onlooker_count = onlooker_count
        self.max_acceleration = max_acceleration
        self.trial_limit = trial_limit
        self.fitness = fitness
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(self.pop_size >= 2, "ABC needs pop_size >= 2")
        check(self.onlooker_count is None or is_int(self.onlooker_count, 0),
              "onlooker_count must be a non negative integer or None")
        check(is_real(self.max_acceleration, 0) and self.max_acceleration > 0, "max_acceleration must be positive")
        check(self.trial_limit is None or is_int(self.trial_limit, 1), "trial_limit must be a positive integer or None")
        check(self.fitness in FITNESS_MODES, f"fitness must be one of {FITNESS_MODES}")

    def initialize(self):
        self.pop = self.random_population()
        self.trials = np.zeros(self.pop_size, dtype=int)
        self._onlookers = self.pop_size if self.onlooker_count is None else self.onlooker_count
        # the paper's limit is SN * D (onlookers = SN). Without onlookers, use SN * D directly
        self._limit = self.trial_limit or max((self._onlookers or self.pop_size) * self.dim, 1)

    def probs(self):
        """Onlooker selection probabilities of the food sources."""
        if self.fitness == "rank" or (self.fitness == "auto" and self.constrained):
            return self.selection_probs(self.pop)
        k = self.min_keys(self.pop)
        fit = np.where(k >= 0, 1 / (1 + np.maximum(k, 0)), 1 - np.minimum(k, 0))
        total = fit.sum()
        if not (np.isfinite(total) and total > 0):
            return self.selection_probs(self.pop)
        return fit / total

    def search(self, src):
        """Change one random component of each source in ``src`` (an index array)."""
        n, m = len(self.pop), len(src)
        k = self.rng.integers(n - 1, size=m)
        k += k >= src  # a random partner other than the source itself
        j = self.rng.integers(self.dim, size=m)
        phi = self.rng.uniform(-self.max_acceleration, self.max_acceleration, m)
        X = self.pop.X[src].copy()
        rows = np.arange(m)
        X[rows, j] += phi * (X[rows, j] - self.pop.X[k, j])
        return self.evaluate(X)

    def iterate(self):
        n = len(self.pop)

        # employed bees
        src = np.arange(n)
        new = self.search(src)
        improved = self.better(new, self.pop)
        self.pop.replace(improved, new)
        self.trials = np.where(improved, 0, self.trials + 1)

        # onlooker bees: each source keeps the best candidate its onlookers found
        if self._onlookers:
            src = self.rng.choice(n, size=self._onlookers, p=self.probs())
            new = self.search(src)
            order = self.argsort(new)
            visited, first = np.unique(src[order], return_index=True)
            best = new[order[first]]
            improved = self.better(best, self.pop[visited])
            rows = visited[improved]
            self.pop.X[rows], self.pop.F[rows], self.pop.V[rows] = best.X[improved], best.F[improved], best.V[improved]
            self.trials += np.bincount(src, minlength=n)
            self.trials[rows] = 0

        # scout bee: at most one per iteration
        i = int(np.argmax(self.trials))
        if self.trials[i] >= self._limit:
            scout = self.random_population(1)
            self.pop.X[i], self.pop.F[i], self.pop.V[i] = scout.X[0], scout.F[0], scout.V[0]
            self.trials[i] = 0
