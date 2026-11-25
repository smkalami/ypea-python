"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real


def reflect(X):
    """Fold points back into [0, 1] by mirroring at the bounds (period 2)."""
    Y = np.mod(X, 2.0)
    return np.where(Y > 1.0, 2.0 - Y, Y)


class CMAES(Algorithm):
    """CMA-ES with rank-mu and rank-one updates and cumulative step size adaptation.

    Parameters
    ----------
    pop_size : number of samples per iteration (lambda). None (default) gives the standard
        4 + floor(3 ln n), where n is the number of codes.
    mu : number of parents. None (default) gives pop_size // 2.
    sigma0 : initial step size, in code units (the search space is [0, 1] per code).
    x0 : initial mean as a code vector in [0, 1]. None (default) draws it at random.

    Samples that leave [0, 1] are reflected back at the bounds, and the reflected points are
    used both for evaluation and for the update. The default population size, weights and
    learning rates follow Table 1 of the tutorial (equations 48 to 58), with positive weights
    only (no active CMA, so the mu best samples are recombined, as in Appendix C). The step
    size is capped at 1, the width of the search space.

    Reference: N. Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772 (2016).
    """

    name = "Covariance Matrix Adaptation Evolution Strategy"
    short_name = "CMA-ES"
    auto_pop_size = True

    def __init__(self, *, pop_size=None, mu=None, sigma0=0.3, x0=None, **kwargs):
        self.mu = mu
        self.sigma0 = sigma0
        self.x0 = x0
        super().__init__(pop_size=pop_size, **kwargs)

    @property
    def lam(self) -> int:
        """Population size in effect (lambda)."""
        n = self.dim
        return self.pop_size if self.pop_size is not None else 4 + int(3 * np.log(n))

    def validate(self):
        super().validate()
        check(self.mu is None or is_int(self.mu, 1), "mu must be a positive integer or None")
        check(self.pop_size is None or self.pop_size >= 2, "pop_size must be at least 2")
        check(self.mu is None or self.pop_size is None or self.mu <= self.pop_size, "mu must not exceed pop_size")
        check(is_real(self.sigma0, 0) and self.sigma0 > 0, "sigma0 must be positive")

    def initialize(self):
        n, lam = self.dim, self.lam
        mu = self.mu if self.mu is not None else lam // 2
        check(mu <= lam, "mu must not exceed the population size")
        # eq. (49): ln((lambda + 1) / 2) - ln i, or ln(mu + 1/2) for a user given mu
        top = np.log((lam + 1) / 2) if self.mu is None else np.log(mu + 0.5)
        w = top - np.log(np.arange(1, mu + 1))
        self.weights = w / w.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        me = self.mu_eff
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (0.25 + me + 1 / me - 2) / ((n + 2) ** 2 + me))  # eq. (58)
        self.damps = 1 + 2 * max(0.0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        if self.x0 is None:
            self.mean = self.rng.random(n)
        else:
            x0 = np.asarray(self.x0, float)
            check(x0.shape == (n,), f"x0 must be a code vector of length {n}, got shape {x0.shape}")
            self.mean = np.clip(x0, 0, 1)
        self.sigma = float(self.sigma0)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.pop = self.evaluate(self.mean[None, :])  # the initial mean, so a best exists

    def iterate(self):
        n, lam = self.dim, self.lam
        mu = len(self.weights)
        Z = self.rng.standard_normal((lam, n))
        X = reflect(self.mean + self.sigma * (Z * self.D) @ self.B.T)
        self.pop = self.evaluate(X)
        order = self.argsort(self.pop)[:mu]
        Y = (self.pop.X[order] - self.mean) / self.sigma  # steps actually taken (after reflection)
        y_w = self.weights @ Y
        self.mean = self.mean + self.sigma * y_w

        # C^(-1/2) y_w = B D^-1 B^T y_w
        c_inv_sqrt_y = self.B @ ((self.B.T @ y_w) / self.D)
        cs, cc = self.cs, self.cc
        self.ps = (1 - cs) * self.ps + np.sqrt(cs * (2 - cs) * self.mu_eff) * c_inv_sqrt_y
        ps_norm = np.linalg.norm(self.ps)
        hsig = ps_norm / np.sqrt(1 - (1 - cs) ** (2 * self.iter)) / self.chi_n < 1.4 + 2 / (n + 1)
        self.pc = (1 - cc) * self.pc + hsig * np.sqrt(cc * (2 - cc) * self.mu_eff) * y_w

        rank_mu = (self.weights[:, None] * Y).T @ Y
        delta = (1 - hsig) * cc * (2 - cc)
        self.C = ((1 - self.c1 - self.cmu + self.c1 * delta) * self.C
                  + self.c1 * np.outer(self.pc, self.pc) + self.cmu * rank_mu)
        self.sigma *= np.exp((cs / self.damps) * (ps_norm / self.chi_n - 1))
        self.sigma = min(self.sigma, 1.0)  # the whole search space is [0, 1]

        self.C = (self.C + self.C.T) / 2
        eigvals, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(eigvals, 1e-20))
