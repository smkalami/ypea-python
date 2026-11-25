"""Invasive Weed Optimization (IWO)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real

FITNESS_MODES = ("auto", "paper", "rank")


class IWO(Algorithm):
    """Invasive Weed Optimization (Mehrabian and Lucas, 2006).

    Parameters
    ----------
    pop_size : maximum population size (competitive exclusion keeps the best ones)
    initial_pop_size : number of weeds at the start
    min_seed_count, max_seed_count : seeds produced by the worst and the best weed
    step_size : initial standard deviation of seed dispersal (code units)
    final_step_size : standard deviation at the end of the run
    modulation_index : exponent of the nonlinear step size schedule
    fitness : "auto" (default), "paper" or "rank". The paper gives each weed a seed count
        that grows linearly from the worst to the best objective value. "auto" uses that
        formula, but switches to ranks when the problem has constraints, because an
        infeasible solution has no meaningful objective value.

    The step size is (1 - u) ** modulation_index * (step_size - final_step_size)
    + final_step_size, where u goes from 0 to 1 over the run (``progress()``: the share of
    ``max_nfe``, ``max_iter`` or ``max_time`` used, whichever is largest). So the step
    reaches final_step_size when the run ends, however it is limited. Weeds with an
    infinite or NaN objective value get min_seed_count seeds.
    """

    name = "Invasive Weed Optimization"
    short_name = "IWO"

    def __init__(self, *, pop_size=20, initial_pop_size=10, min_seed_count=0, max_seed_count=5,
                 step_size=0.1, final_step_size=0.001, modulation_index=2.0, fitness="auto", **kwargs):
        self.initial_pop_size = initial_pop_size
        self.min_seed_count, self.max_seed_count = min_seed_count, max_seed_count
        self.step_size, self.final_step_size = step_size, final_step_size
        self.modulation_index = modulation_index
        self.fitness = fitness
        super().__init__(pop_size=pop_size, **kwargs)

    def validate(self):
        super().validate()
        check(is_int(self.initial_pop_size, 1), "initial_pop_size must be a positive integer")
        check(is_int(self.min_seed_count, 0) and is_int(self.max_seed_count, 1)
              and self.min_seed_count <= self.max_seed_count, "need 0 <= min_seed_count <= max_seed_count, max >= 1")
        check(is_real(self.step_size, 0) and is_real(self.final_step_size, 0), "step sizes must be non negative")
        check(is_real(self.modulation_index, 0), "modulation_index must be non negative")
        check(self.fitness in FITNESS_MODES, f"fitness must be one of {FITNESS_MODES}")

    def initialize(self):
        self.pop = self.random_population(self.initial_pop_size)

    def seed_ratio(self):
        """For each weed, a number in [0, 1]: 0 for the worst weed, 1 for the best."""
        use_ranks = self.fitness == "rank" or (self.fitness == "auto" and self.constrained)
        n = len(self.pop)
        if use_ranks:
            return 1 - self.ranks(self.pop) / max(n - 1, 1)
        k = self.min_keys(self.pop)
        finite = np.isfinite(k)
        if not finite.any():
            return np.ones(n)
        best, worst = k[finite].min(), k[finite].max()
        if worst == best:
            return np.where(finite, 1.0, 0.0)
        # weeds with infinite or NaN values get no seeds
        return np.where(finite, (worst - np.where(finite, k, worst)) / (worst - best), 0.0)

    def step(self):
        frac = 1.0 - self.progress()
        return frac**self.modulation_index * (self.step_size - self.final_step_size) + self.final_step_size

    def iterate(self):
        smin, smax = self.min_seed_count, self.max_seed_count
        counts = np.floor(smin + (smax - smin) * self.seed_ratio()).astype(int)
        parents = np.repeat(np.arange(len(self.pop)), counts)
        if len(parents):
            X = self.pop.X[parents] + self.step() * self.rng.standard_normal((len(parents), self.dim))
            merged = self.pop.concat(self.pop, self.evaluate(X))
        else:
            merged = self.pop
        self.pop = self.sort(merged)[: self.pop_size]
