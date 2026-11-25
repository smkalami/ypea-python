"""Harmony Search (HS)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real


class HS(Algorithm):
    """Harmony Search (Geem, Kim and Loganathan, 2001), continuous version with a
    bandwidth (Lee and Geem, 2005).

    Each iteration creates ``new_count`` new harmonies. For every component of a new
    harmony, with probability ``hmcr`` the value is copied from a random member of the
    harmony memory (a new member is drawn for each component). A copied value is then,
    with probability ``par``, moved by ``bandwidth * U(-1, 1)``, as in the paper.
    Otherwise the component is random in [0, 1]. The new harmonies join the memory, and
    the best ``pop_size`` harmonies are kept.

    Parameters
    ----------
    pop_size : harmony memory size (HMS), 10 by default. ``hms`` is an alias.
    new_count : number of new harmonies per iteration. The original algorithm creates
        one harmony per iteration and replaces the worst one if it is better, which is
        the case ``new_count=1``.
    hmcr : harmony memory consideration rate
    par : pitch adjustment rate
    fret_width : initial bandwidth (in code units, where each variable spans [0, 1])
    fret_width_damp : the bandwidth is multiplied by this factor after every iteration.
        The paper keeps the bandwidth fixed (``fret_width_damp=1``).

    References
    ----------
    Z. W. Geem, J. H. Kim and G. V. Loganathan, "A new heuristic optimization algorithm:
    Harmony search", Simulation 76(2), 2001.
    K. S. Lee and Z. W. Geem, "A new meta-heuristic algorithm for continuous engineering
    optimization: harmony search theory and practice", Computer Methods in Applied
    Mechanics and Engineering 194, 2005.
    """

    name = "Harmony Search"
    short_name = "HS"

    def __init__(self, *, pop_size=10, new_count=20, hmcr=0.9, par=0.1, fret_width=0.02,
                 fret_width_damp=0.99, **kwargs):
        self.new_count = new_count
        self.hmcr = hmcr
        self.par = par
        self.fret_width = fret_width
        self.fret_width_damp = fret_width_damp
        super().__init__(pop_size=pop_size, **kwargs)

    @property
    def hms(self):
        """Harmony memory size (alias of ``pop_size``)."""
        return self.pop_size

    @hms.setter
    def hms(self, value):
        self.pop_size = value
        self.validate()

    def validate(self):
        super().validate()
        check(is_int(self.new_count, 1), "new_count must be a positive integer")
        check(is_real(self.hmcr, 0, 1), "hmcr must be in [0, 1]")
        check(is_real(self.par, 0, 1), "par must be in [0, 1]")
        check(is_real(self.fret_width, 0), "fret_width must be non negative")
        check(is_real(self.fret_width_damp, 0) and self.fret_width_damp > 0, "fret_width_damp must be positive")

    def initialize(self):
        self.pop = self.sort(self.random_population())
        self.bandwidth = self.fret_width

    def iterate(self):
        n, d = self.new_count, self.dim
        rng = self.rng
        X = rng.random((n, d))
        use_memory = rng.random((n, d)) < self.hmcr
        members = rng.integers(len(self.pop), size=(n, d))
        X[use_memory] = self.pop.X[members, np.arange(d)][use_memory]
        adjust = use_memory & (rng.random((n, d)) < self.par)
        X[adjust] += self.bandwidth * rng.uniform(-1, 1, adjust.sum())
        merged = self.pop.concat(self.pop, self.evaluate(X))
        self.pop = self.sort(merged)[: self.pop_size]
        self.bandwidth *= self.fret_width_damp
