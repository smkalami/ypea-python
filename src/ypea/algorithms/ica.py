"""Imperialist Competitive Algorithm (ICA)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real


class ICA(Algorithm):
    """Imperialist Competitive Algorithm (Atashpaz-Gargari and Lucas, 2007).

    The population is split into countries. The best ``empire_count`` countries are
    imperialists, and the others are colonies shared among them. In every iteration the
    colonies move toward their imperialist (assimilation), some countries change at random
    (revolution), a colony that becomes better than its imperialist takes its place, and
    the empires compete: the weakest colony of the weakest empire goes to another empire.
    When the weakest empire has no colonies left, it collapses, and its imperialist becomes
    a colony of another empire (only the weakest empire is checked, as in MATLAB, so an
    empty empire lasts until it is the weakest). With one empire left, the run continues
    with assimilation and revolution.

    Parameters
    ----------
    pop_size : number of countries (imperialists and colonies)
    empire_count : number of empires (imperialists) at the start, at least 2 and less
        than ``pop_size``. The number of colonies is ``pop_size - empire_count``.
    selection_pressure : pressure of the power based probabilities (see below)
    assimilation_coeff : beta. Each colony component moves by beta * U(0, 1) times its
        distance to the imperialist.
    revolution_prob : probability that a colony is revolved in an iteration
    revolution_rate : fraction of the code components changed by a revolution
    revolution_step_size : initial standard deviation of the revolution (code units)
    revolution_step_size_damp : the step size is multiplied by this after every iteration
    zeta : weight of the mean colony cost in the total cost of an empire

    Rank based costs
    ----------------
    Every cost is the normalized rank of a country in the whole population,
    c = rank / (pop_size - 1), from 0 (best) to 1 (worst), so the method works for
    maximization, any objective scale, and constraints. The total cost of empire k is
    TC_k = c(imperialist) + zeta * mean(c(colonies)), or c(imperialist) without colonies.
    Probabilities are p_k proportional to exp(-selection_pressure * x_k / max(x)),
    where x is the imperialist costs (initial split) or the total costs (competition).
    In the competition the weakest empire gets p = 0.

    Differences from the paper
    --------------------------
    * The paper uses a linear power (cost minus the largest cost) on raw objective
      values. Here the power is exponential in rank based costs, as in YPEA for MATLAB,
      which keeps ``selection_pressure`` meaningful.
    * Assimilation draws an independent U(0, beta) factor per component, as in YPEA for
      MATLAB, instead of a step along the line to the imperialist plus a random angle
      theta. The per component factors already deviate the move from the direct line,
      and an angle is not well defined in many dimensions.
    * Revolution is not in the 2007 paper (it comes from later versions of ICA). As in
      YPEA for MATLAB, each imperialist tries one Gaussian revolution per iteration and
      keeps it only if it is better, and each colony is revolved with probability
      ``revolution_prob`` and always keeps the change.
    * A revolved colony is revolved right after assimilation and evaluated once.
    * The paper stops when one empire is left. Here the run continues until a limit.

    Reference: E. Atashpaz-Gargari and C. Lucas, "Imperialist competitive algorithm:
    an algorithm for optimization inspired by imperialistic competition," IEEE Congress
    on Evolutionary Computation, 2007, pp. 4661-4667.
    """

    name = "Imperialist Competitive Algorithm"
    short_name = "ICA"

    def __init__(self, *, empire_count=5, selection_pressure=1.0, assimilation_coeff=1.5,
                 revolution_prob=0.05, revolution_rate=0.1, revolution_step_size=0.1,
                 revolution_step_size_damp=0.99, zeta=0.2, **kwargs):
        self.empire_count = empire_count
        self.selection_pressure = selection_pressure
        self.assimilation_coeff = assimilation_coeff
        self.revolution_prob = revolution_prob
        self.revolution_rate = revolution_rate
        self.revolution_step_size = revolution_step_size
        self.revolution_step_size_damp = revolution_step_size_damp
        self.zeta = zeta
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_int(self.empire_count, 2) and self.empire_count < self.pop_size,
              "empire_count must be an integer, at least 2 and less than pop_size")
        check(is_real(self.selection_pressure, 0), "selection_pressure must be non negative")
        check(is_real(self.assimilation_coeff, 0), "assimilation_coeff must be non negative")
        check(is_real(self.revolution_prob, 0, 1), "revolution_prob must be in [0, 1]")
        check(is_real(self.revolution_rate, 0, 1), "revolution_rate must be in [0, 1]")
        check(is_real(self.revolution_step_size, 0), "revolution_step_size must be non negative")
        check(is_real(self.revolution_step_size_damp, 0) and self.revolution_step_size_damp > 0,
              "revolution_step_size_damp must be positive")
        check(is_real(self.zeta, 0), "zeta must be non negative")

    @property
    def colony_count(self):
        return self.pop_size - self.empire_count

    # Empires are two arrays: ``empire_of[i]`` is the empire of country i, and
    # ``imperialists[k]`` is the country that rules empire k (empire_of[imperialists[k]] == k).

    def costs(self):
        """Rank based cost of every country, 0 (best) to 1 (worst)."""
        return self.ranks(self.pop) / max(len(self.pop) - 1, 1)

    def probs(self, x):
        top = x.max()
        p = np.exp(-self.selection_pressure * (x / top if top > 0 else x))
        return p / p.sum()

    def colony_counts(self):
        return np.bincount(self.empire_of, minlength=len(self.imperialists)) - 1

    def total_costs(self, c):
        k = len(self.imperialists)
        counts = self.colony_counts()
        col_sum = np.bincount(self.empire_of, weights=c, minlength=k) - c[self.imperialists]
        mean_col = np.divide(col_sum, counts, out=np.zeros(k), where=counts > 0)
        return c[self.imperialists] + self.zeta * mean_col

    def initialize(self):
        self.pop = self.sort(self.random_population())
        k = self.empire_count
        self.imperialists = np.arange(k)
        # the paper's deterministic split: colony counts proportional to power,
        # rounded by largest remainder so they add up to colony_count
        share = self.probs(self.costs()[:k]) * self.colony_count
        counts = np.floor(share).astype(int)
        extra = self.colony_count - counts.sum()
        counts[np.argsort(counts - share, kind="stable")[:extra]] += 1
        self.empire_of = np.empty(self.pop_size, dtype=int)
        self.empire_of[:k] = self.imperialists
        self.empire_of[k + self.rng.permutation(self.colony_count)] = np.repeat(np.arange(k), counts)
        self._sigma = self.revolution_step_size

    def revolve(self, X):
        n, d = X.shape
        m = int(np.ceil(self.revolution_rate * d))
        cols = np.argsort(self.rng.random((n, d)), axis=1)[:, :m]
        Y = X.copy()
        Y[np.arange(n)[:, None], cols] += self._sigma * self.rng.standard_normal((n, m))
        return Y

    def iterate(self):
        n = len(self.pop)
        imps = self.imperialists
        ruler = imps[self.empire_of]
        colony = ruler != np.arange(n)

        # assimilation (imperialists do not move, since their distance is zero)
        X = self.pop.X
        Xc = X + self.assimilation_coeff * self.rng.random(X.shape) * (X[ruler] - X)
        rev = colony & (self.rng.random(n) < self.revolution_prob)
        Xc[rev] = self.revolve(Xc[rev])
        trial_imp = self.revolve(X[imps])

        new = self.evaluate(np.vstack([Xc[colony], trial_imp]))
        nc = int(colony.sum())
        _set_rows(self.pop, np.flatnonzero(colony), new, np.arange(nc))
        keep = self.better(new[nc:], self.pop[imps])
        _set_rows(self.pop, imps[keep], new, nc + np.flatnonzero(keep))

        # exchange: the best country of each empire becomes its imperialist
        r = self.ranks(self.pop)
        for k in range(len(imps)):
            members = np.flatnonzero(self.empire_of == k)
            best = members[np.argmin(r[members])]
            if r[best] < r[imps[k]]:
                imps[k] = best

        self.compete(r / max(n - 1, 1))
        self._sigma *= self.revolution_step_size_damp

    def compete(self, c):
        """Imperialistic competition and collapse of empires without colonies."""
        k = len(self.imperialists)
        if k == 1:
            return
        tc = self.total_costs(c)
        weakest = int(np.argmax(tc))
        p = self.probs(tc)
        p[weakest] = 0
        p /= p.sum()
        members = np.flatnonzero(self.empire_of == weakest)
        colonies = members[members != self.imperialists[weakest]]
        if len(colonies):
            self.empire_of[colonies[np.argmax(c[colonies])]] = self.rng.choice(k, p=p)
        if len(colonies) <= 1:
            # the weakest empire has no colonies left: its imperialist joins another empire
            self.empire_of[self.imperialists[weakest]] = self.rng.choice(k, p=p)
            self.imperialists = np.delete(self.imperialists, weakest)
            self.empire_of[self.empire_of > weakest] -= 1


def _set_rows(pop, idx, src, rows):
    """Copy rows ``rows`` of ``src`` into rows ``idx`` of ``pop`` (in place)."""
    # index the arrays directly: Population cannot hold zero rows with a reshape
    pop.X[idx] = src.X[rows]
    pop.F[idx] = src.F[rows]
    pop.V[idx] = src.V[rows]
