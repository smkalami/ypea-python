"""Firefly Algorithm (FA)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_real


class FA(Algorithm):
    """Firefly Algorithm (Yang, 2009 and 2010).

    Every firefly moves toward each brighter (better) firefly in turn. The attractiveness
    of firefly j seen from firefly i is beta = beta_base * exp(-gamma * r ** m), where r is
    their distance in code space divided by sqrt(d), the largest possible distance. One
    move is

        x_i = x_i + beta * U(0, 1) * (x_j - x_i) + alpha * delta * U(-1, 1)

    with U drawn for every component. The brightest firefly has no brighter neighbor and
    makes a random walk, x_i = x_i + alpha * delta * U(-1, 1). Brightness is the rank
    given by the comparator at the start of the iteration, so maximization and
    constraints work. After all moves, each firefly is evaluated once, so an iteration
    costs ``pop_size`` evaluations. The new fireflies are merged with the old ones and
    the best ``pop_size`` are kept. ``alpha`` is multiplied by ``alpha_damp`` after every
    iteration.

    Parameters
    ----------
    gamma : light absorption coefficient
    beta_base : attractiveness at distance zero
    alpha : initial randomization (mutation) coefficient
    alpha_damp : damping factor of alpha per iteration
    delta : scale of the random step (code units)
    m : exponent of the distance in the attractiveness

    Differences from the paper: the distance is divided by sqrt(d) (eq. 8 uses the plain
    Cartesian distance), and the random factor U(0, 1) on the attraction term, the
    merge with the old population and the ``delta`` scale follow YPEA for MATLAB. With
    the default beta_base = 2, the mean attraction is 1, as with beta_0 = 1 in the paper.
    As in Yang's reference code, the light intensities are updated once per iteration,
    not after every single move. The random term alpha (rand - 1/2) of eq. 9 becomes
    ``alpha * delta * U(-1, 1)`` in code units, and ``alpha_damp`` shrinks it, which the
    paper only suggests as future work.

    References: X.-S. Yang, "Firefly algorithms for multimodal optimization", Stochastic
    Algorithms: Foundations and Applications (SAGA 2009), LNCS 5792, pp. 169 to 178.
    X.-S. Yang, "Nature-Inspired Metaheuristic Algorithms", 2nd edition, Luniver Press, 2010.
    """

    name = "Firefly Algorithm"
    short_name = "FA"

    def __init__(self, *, gamma=1.0, beta_base=2.0, alpha=0.2, alpha_damp=0.99, delta=0.05, m=2.0, **kwargs):
        self.gamma = gamma
        self.beta_base = beta_base
        self.alpha = alpha
        self.alpha_damp = alpha_damp
        self.delta = delta
        self.m = m
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        for name in ("beta_base", "alpha", "alpha_damp", "delta"):
            check(is_real(getattr(self, name)) and getattr(self, name) > 0, f"{name} must be positive")
        check(is_real(self.gamma, 0), "gamma must be non negative")
        check(is_real(self.m, 1), "m must be at least 1")

    def initialize(self):
        self.pop = self.random_population()
        self._alpha = self.alpha

    def iterate(self):
        pop = self.sort(self.pop)
        old = pop.X
        n, d = old.shape
        rank = self.ranks(pop)
        X = old.copy()
        step = self._alpha * self.delta
        # firefly j (in order from the brightest) attracts every dimmer firefly i
        for j in range(n):
            i = rank > rank[j]
            if not i.any():
                continue
            diff = old[j] - X[i]
            r = np.linalg.norm(diff, axis=1, keepdims=True) / np.sqrt(d)
            beta = self.beta_base * np.exp(-self.gamma * r**self.m)
            X[i] += beta * self.rng.random(diff.shape) * diff + step * self.rng.uniform(-1, 1, diff.shape)
        # the brightest fireflies (no brighter neighbor) make a random walk
        top = rank == 0
        X[top] += step * self.rng.uniform(-1, 1, (int(top.sum()), d))
        new = self.evaluate(X)
        self.pop = self.sort(pop.concat(pop, new))[:n]
        self._alpha *= self.alpha_damp
