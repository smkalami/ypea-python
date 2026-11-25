"""Differential Evolution (DE)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real

_BASES = {
    "rand": "rand", "best": "best",
    "current-to-best": "current-to-best", "target-to-best": "current-to-best",
    "rand-to-best": "rand-to-best",
}
_CROSSOVERS = {"bin": "bin", "binomial": "bin", "exp": "exp", "exponential": "exp"}


def parse_strategy(text):
    """Parse 'DE/base/k/crossover'. Missing parts default to 'DE/rand/1/bin'."""
    parts = [p for p in str(text).replace("\\", "/").replace(" ", "").lower().split("/") if p]
    if parts and parts[0] == "de":
        parts = parts[1:]
    base, k, cx = "rand", 1, "bin"
    for p in parts:
        if p.isdigit():
            k = int(p)
        elif p in _BASES:
            base = _BASES[p]
        elif p in _CROSSOVERS:
            cx = _CROSSOVERS[p]
        else:
            raise ValueError(f"unknown part {p!r} in DE strategy {text!r}")
    return base, k, cx


class DE(Algorithm):
    """Differential Evolution (Storn and Price, 1997).

    Parameters
    ----------
    strategy : str, optional
        Short form like ``"DE/best/2/bin"``. Sets ``base_vector``, ``diff_vectors_count``
        and ``crossover_method``.
    base_vector : "rand", "best", "current-to-best" or "rand-to-best"
        ("target-to-best" is accepted as the MATLAB name of "current-to-best").
    diff_vectors_count : number of difference vectors
    crossover_method : "bin" or "exp"
    crossover_prob : crossover rate CR
    beta_min, beta_max : the scale factor is drawn uniformly from [beta_min, beta_max]
        for every component (dither), as in YPEA for MATLAB.

    A trial vector replaces its target when it is at least as good (as in the paper).
    All trial vectors of an iteration are built from the same population (synchronous).
    """

    name = "Differential Evolution"
    short_name = "DE"

    def __init__(self, strategy=None, *, base_vector="rand", diff_vectors_count=1,
                 crossover_method="bin", crossover_prob=0.2, beta_min=0.2, beta_max=0.9, **kwargs):
        if strategy is not None:
            base_vector, diff_vectors_count, crossover_method = parse_strategy(strategy)
        self.base_vector = _BASES.get(base_vector, base_vector)
        self.diff_vectors_count = diff_vectors_count
        self.crossover_method = _CROSSOVERS.get(crossover_method, crossover_method)
        self.crossover_prob = crossover_prob
        self.beta_min, self.beta_max = beta_min, beta_max
        super().__init__(**kwargs)

    @property
    def strategy(self):
        return f"DE/{self.base_vector}/{self.diff_vectors_count}/{self.crossover_method}"

    def validate(self):
        super().validate()
        check(self.base_vector in _BASES.values(), f"unknown base_vector {self.base_vector!r}")
        check(self.crossover_method in ("bin", "exp"), "crossover_method must be 'bin' or 'exp'")
        check(is_int(self.diff_vectors_count, 1), "diff_vectors_count must be a positive integer")
        check(is_real(self.crossover_prob, 0, 1), "crossover_prob must be in [0, 1]")
        check(is_real(self.beta_min, 0) and is_real(self.beta_max, 0) and self.beta_min <= self.beta_max,
              "need 0 <= beta_min <= beta_max")
        need = 2 * self.diff_vectors_count + (self.base_vector in ("rand", "rand-to-best")) + 1
        check(self.pop_size >= need, f"{self.strategy} needs pop_size >= {need}")

    def initialize(self):
        self.pop = self.random_population()

    def _beta(self, shape):
        return self.rng.uniform(self.beta_min, self.beta_max, shape)

    def iterate(self):
        X = self.pop.X
        n, d = X.shape
        k = self.diff_vectors_count
        uses_rand = self.base_vector in ("rand", "rand-to-best")
        # distinct random partners for every target, none equal to the target itself
        keys = self.rng.random((n, n))
        np.fill_diagonal(keys, np.inf)
        idx = np.argsort(keys, axis=1)[:, : 2 * k + uses_rand]

        best = self.best.X
        if self.base_vector == "rand":
            base = X[idx[:, 0]]
        elif self.base_vector == "best":
            base = np.broadcast_to(best, X.shape)
        elif self.base_vector == "current-to-best":
            base = X + self._beta(X.shape) * (best - X)
        else:  # rand-to-best
            r = X[idx[:, 0]]
            base = r + self._beta(X.shape) * (best - r)
        diffs = idx[:, int(uses_rand):]
        mutant = base.copy()
        for j in range(k):
            mutant += self._beta(X.shape) * (X[diffs[:, 2 * j]] - X[diffs[:, 2 * j + 1]])

        mask = self._crossover_mask(n, d)
        trial = np.where(mask, mutant, X)
        new = self.evaluate(trial)
        self.pop.replace(self.no_worse(new, self.pop), new)

    def _crossover_mask(self, n, d):
        cr = self.crossover_prob
        if self.crossover_method == "bin":
            mask = self.rng.random((n, d)) <= cr
            mask[np.arange(n), self.rng.integers(d, size=n)] = True
            return mask
        # exponential: from a random start, copy L consecutive components (cyclic), where
        # L is 1 plus the number of successes before the first failure, at most d
        start = self.rng.integers(d, size=n)
        if cr >= 1:
            length = np.full(n, d)
        else:
            length = np.minimum(self.rng.geometric(1 - cr, size=n), d)
        offset = (np.arange(d)[None, :] - start[:, None]) % d
        return offset < length[:, None]
