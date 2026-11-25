"""Particle Swarm Optimization (PSO)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_real


class PSO(Algorithm):
    """Particle Swarm Optimization with inertia weight (global best topology).

    Parameters
    ----------
    w : inertia weight
    wdamp : the inertia weight is multiplied by this after every iteration
    c1 : personal acceleration coefficient
    c2 : global acceleration coefficient

    Particles that leave [0, 1] are clipped and the offending velocity components are
    reversed (as in YPEA for MATLAB). Use ``use_constriction_coeffs`` for the
    Clerc and Kennedy constriction coefficients.

    References: Kennedy and Eberhart (1995). Shi and Eberhart (1998).
    Clerc and Kennedy (2002).
    """

    name = "Particle Swarm Optimization"
    short_name = "PSO"

    def __init__(self, *, w=1.0, wdamp=0.99, c1=2.0, c2=2.0, **kwargs):
        self.w, self.wdamp, self.c1, self.c2 = w, wdamp, c1, c2
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_real(self.w, 0), "w must be non negative")
        check(is_real(self.wdamp, 0, 1) and self.wdamp > 0, "wdamp must be in (0, 1]")
        check(is_real(self.c1, 0) and is_real(self.c2, 0), "c1 and c2 must be non negative")

    def use_constriction_coeffs(self, phi1=2.05, phi2=2.05):
        """Set w, c1 and c2 from Clerc and Kennedy (2002). phi1 + phi2 is raised to 4 if lower."""
        phi = phi1 + phi2
        if phi < 4:
            phi1, phi2, phi = 4 * phi1 / phi, 4 * phi2 / phi, 4.0
        chi = float(2 / (phi - 2 + np.sqrt(phi**2 - 4 * phi)))
        self.w, self.wdamp, self.c1, self.c2 = chi, 1.0, chi * phi1, chi * phi2
        return self

    def initialize(self):
        self.pop = self.random_population()
        self.velocity = np.zeros_like(self.pop.X)
        self.pbest = self.pop.copy()
        self._w = self.w

    def iterate(self):
        X, V = self.pop.X, self.velocity
        r1, r2 = self.rng.random((2, *X.shape))
        V = self._w * V + self.c1 * r1 * (self.pbest.X - X) + self.c2 * r2 * (self.best.X - X)
        X = X + V
        out = (X < 0) | (X > 1)
        V[out] = -V[out]
        self.velocity = V
        self.pop = self.evaluate(X)
        self.pbest.replace(self.better(self.pop, self.pbest), self.pop)
        self._w *= self.wdamp
