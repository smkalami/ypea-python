"""Random Search."""

from __future__ import annotations

from ..algorithm import Algorithm


class RandomSearch(Algorithm):
    """Pure random search: every iteration samples ``pop_size`` new points uniformly.

    A reference method for comparisons. It keeps no state, because the framework tracks
    the best solution. Each iteration makes exactly one draw, ``rng.random((pop_size, d))``.

    Parameters
    ----------
    pop_size : number of random points per iteration (default 100)
    """

    name = "Random Search"
    short_name = "RS"

    def initialize(self):
        self.pop = self.random_population()

    def iterate(self):
        self.pop = self.random_population()
