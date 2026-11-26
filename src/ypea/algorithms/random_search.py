"""Random Search."""

from __future__ import annotations

from ..algorithm import Algorithm


class RandomSearch(Algorithm):
    """Pure random search: every iteration samples ``pop_size`` new points uniformly.

    A reference method for comparisons. It keeps no state, because the framework tracks
    the best solution (or, with several objectives, the front). Each iteration makes
    exactly one draw, ``rng.random((pop_size, d))``. It solves single and multi-objective
    problems.

    Parameters
    ----------
    pop_size : number of random points per iteration (default 100)
    max_archive : int or None
        With several objectives, the largest size of the front. When more non-dominated
        solutions are found, the most crowded ones are dropped. None (the default) keeps
        all of them.
    """

    name = "Random Search"
    short_name = "RS"
    objectives = "both"

    def __init__(self, *, max_archive=None, **kwargs):
        self.max_archive = max_archive
        super().__init__(**kwargs)

    def initialize(self):
        self.pop = self.random_population()

    def iterate(self):
        self.pop = self.random_population()
