"""Biogeography-based Optimization (BBO)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_real


class BBO(Algorithm):
    """Biogeography-based Optimization (Simon, 2008).

    Each solution is a habitat. The habitats are kept sorted from best to worst, and
    their emigration rates mu go linearly from 1 (best) to 0 (worst), with immigration
    rates lambda = 1 - mu. Every feature k of habitat i immigrates with probability
    lambda_i from a habitat j != i chosen by roulette wheel on mu:
    x_ik + alpha * (x_jk - x_ik). Every feature then mutates with probability
    ``mutation_prob`` by a normal step. All new habitats are built from the same
    population (synchronous). The best ``keep_rate`` share of the old habitats is kept,
    and the rest of the population is filled with the best new habitats.

    Parameters
    ----------
    keep_rate : share of the best habitats kept from one iteration to the next (elitism)
    alpha : migration step. 1 (the default) copies the feature as in the paper. YPEA for
        MATLAB used 0.9.
    mutation_prob : probability that a feature mutates
    mutation_step_size : standard deviation of the mutation step (code units)
    mutation_step_size_damp : the step size is multiplied by this after every iteration

    Differences from the paper: Simon mutates a habitat with a probability based on its
    species count and replaces a feature with a random value. This version keeps the
    Gaussian mutation of YPEA for MATLAB, with a fixed probability per feature and a
    shrinking step size. Elitism keeps a share of the population instead of a fixed
    number of habitats.

    Reference: Simon (2008), Biogeography-based optimization, IEEE Transactions on
    Evolutionary Computation 12(6), 702 to 713.
    """

    name = "Biogeography-based Optimization"
    short_name = "BBO"

    def __init__(self, *, keep_rate=0.2, alpha=1.0, mutation_prob=0.1, mutation_step_size=0.05,
                 mutation_step_size_damp=0.99, **kwargs):
        self.keep_rate = keep_rate
        self.alpha = alpha
        self.mutation_prob = mutation_prob
        self.mutation_step_size = mutation_step_size
        self.mutation_step_size_damp = mutation_step_size_damp
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_real(self.keep_rate, 0, 1), "keep_rate must be in [0, 1]")
        check(is_real(self.alpha, 0), "alpha must be non negative")
        check(is_real(self.mutation_prob, 0, 1), "mutation_prob must be in [0, 1]")
        check(is_real(self.mutation_step_size, 0), "mutation_step_size must be non negative")
        check(is_real(self.mutation_step_size_damp, 0) and self.mutation_step_size_damp > 0,
              "mutation_step_size_damp must be positive")

    def initialize(self):
        n = self.pop_size
        self.pop = self.sort(self.random_population())
        # MATLAB round (half away from zero)
        self._keep = int(np.floor(self.keep_rate * n + 0.5))
        self.mu = np.linspace(1, 0, n)
        self.lam = 1 - self.mu
        self._sigma = self.mutation_step_size

    def iterate(self):
        X = self.pop.X
        n, d = X.shape
        Xnew = X.copy()
        if n > 1:
            migrate = self.rng.random((n, d)) <= self.lam[:, None]
            P = np.tile(self.mu, (n, 1))
            np.fill_diagonal(P, 0)
            C = np.cumsum(P, axis=1)
            C /= np.maximum(C[:, -1:], 1e-300)  # a zero row (n = 2, best habitat) never migrates
            u = self.rng.random((n, d))
            src = np.array([np.searchsorted(C[i], u[i], side="right") for i in range(n)])
            src = np.minimum(src, n - 1)  # guards against round off in the last cumulative sum
            Xnew = np.where(migrate, X + self.alpha * (X[src, np.arange(d)] - X), X)
        mutate = self.rng.random((n, d)) <= self.mutation_prob
        Xnew = Xnew + mutate * self._sigma * self.rng.standard_normal((n, d))
        new = self.sort(self.evaluate(Xnew))
        rows = np.r_[np.arange(self._keep), n + np.arange(n - self._keep)]
        self.pop = self.sort(self.pop.concat(self.pop, new)[rows])
        self._sigma *= self.mutation_step_size_damp
