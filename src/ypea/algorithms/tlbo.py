"""Teaching-Learning-Based Optimization (TLBO)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check


class TLBO(Algorithm):
    """Teaching-Learning-Based Optimization (Rao, Savsani and Vakharia, 2011).

    Each iteration has two phases, and each phase evaluates ``pop_size`` new solutions.

    Teacher phase: the teacher is the best learner of the class. Every learner moves by
    ``r * (teacher - TF * mean)``, where ``mean`` is the mean of the class, ``r`` is
    uniform in [0, 1] per component and the teaching factor ``TF`` is 1 or 2 at random
    for each learner.

    Learner phase: every learner i picks a random partner j (not i). It moves by
    ``r * (x_i - x_j)`` (away from j) if it is better than j, otherwise by
    ``r * (x_j - x_i)`` (toward j).

    In both phases a new solution replaces the learner only if it is strictly better.

    TLBO has no settings beyond the common ones (``pop_size`` must be at least 2).

    Differences from the paper: each phase updates all learners at once, from the same
    class (synchronous). The paper describes the phases learner by learner, and an
    asynchronous version lets later learners see the earlier updates. The paper's
    removal of duplicate solutions is not done.

    Reference
    ---------
    R. V. Rao, V. J. Savsani and D. P. Vakharia, "Teaching-learning-based optimization:
    A novel method for constrained mechanical design optimization problems",
    Computer-Aided Design 43(3), 2011.
    """

    name = "Teaching-Learning-Based Optimization"
    short_name = "TLBO"

    def validate(self):
        super().validate()
        check(self.pop_size >= 2, "TLBO needs pop_size >= 2")

    def initialize(self):
        self.pop = self.random_population()

    def _accept(self, X):
        new = self.evaluate(X)
        self.pop.replace(self.better(new, self.pop), new)

    def iterate(self):
        rng = self.rng
        X = self.pop.X
        n, d = X.shape

        # teacher phase
        teacher = X[self.argbest(self.pop)]
        tf = rng.integers(1, 3, size=(n, 1))
        self._accept(X + rng.random((n, d)) * (teacher - tf * X.mean(axis=0)))

        # learner phase (on the class after the teacher phase)
        X = self.pop.X
        j = rng.integers(n - 1, size=n)
        j += j >= np.arange(n)
        partner = self.pop[j]
        step = np.where(self.better(self.pop, partner)[:, None], X - X[j], X[j] - X)
        self._accept(X + rng.random((n, d)) * step)
