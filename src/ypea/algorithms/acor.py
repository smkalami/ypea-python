"""Ant Colony Optimization for Continuous Domains (ACOR)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real


class ACOR(Algorithm):
    """Ant Colony Optimization for Continuous Domains (Socha and Dorigo, 2008).

    The archive holds the ``pop_size`` best solutions found so far (the paper's k),
    sorted from best to worst. Solution l (1 based) has the weight
    w_l = exp(-(l - 1)^2 / (2 q^2 k^2)) / (q k sqrt(2 pi)). Each new solution picks one
    guide l with probability proportional to w_l and samples every component i from a
    normal distribution with mean s_l,i and standard deviation
    zeta * sum_e |s_e,i - s_l,i| / (k - 1). The new solutions are merged with the archive
    and the best k are kept.

    Parameters
    ----------
    pop_size : archive size k
    sample_count : number of new solutions (ants) per iteration (the paper's m)
    q : intensification factor. Small values favor the best solutions.
    zeta : deviation distance ratio (the paper's xi). Larger values slow convergence.

    The defaults are those of YPEA for MATLAB. The paper tuned k = 50, m = 2,
    q = 0.0001 and xi = 0.85 for its benchmark functions.

    Reference: Socha, K. and Dorigo, M. (2008). Ant colony optimization for continuous
    domains. European Journal of Operational Research, 185(3), 1155 to 1173.
    """

    name = "Ant Colony Optimization for Continuous Domains"
    short_name = "ACOR"

    def __init__(self, *, sample_count=50, q=0.5, zeta=1.0, **kwargs):
        self.sample_count = sample_count
        self.q = q
        self.zeta = zeta
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(self.pop_size >= 2, "ACOR needs pop_size >= 2")
        check(is_int(self.sample_count, 1), "sample_count must be a positive integer")
        check(is_real(self.q, 0) and self.q > 0, "q must be positive")
        check(is_real(self.zeta, 0) and self.zeta > 0, "zeta must be positive")

    def initialize(self):
        k = self.pop_size
        self.pop = self.sort(self.random_population())
        qk = self.q * k
        w = np.exp(-0.5 * (np.arange(k) / qk) ** 2) / (qk * np.sqrt(2 * np.pi))
        self.weights = w / w.sum()

    def iterate(self):
        S = self.pop.X
        k = len(S)
        # one guide per new solution, used for all its components
        guide = self.rng.choice(k, size=self.sample_count, p=self.weights)
        mean = S[guide]
        sigma = self.zeta * np.abs(mean[:, None, :] - S[None, :, :]).sum(axis=1) / (k - 1)
        X = mean + sigma * self.rng.standard_normal(mean.shape)
        # archive first, so the stable sort keeps old solutions on ties
        self.pop = self.sort(self.pop.concat(self.pop, self.evaluate(X)))[:k]
