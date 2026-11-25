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

    The step size at iteration t of T is
    ((T - t) / T) ** modulation_index * (step_size - final_step_size) + final_step_size,
    where T is the planned number of iterations.
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
        mean_seeds = (self.min_seed_count + self.max_seed_count) / 2
        self._planned = self.planned_iterations(self.pop_size * max(mean_seeds, 1))

    def seed_ratio(self):
        """For each weed, a number in [0, 1]: 0 for the worst weed, 1 for the best."""
        use_ranks = self.fitness == "rank" or (self.fitness == "auto" and self.constrained)
        n = len(self.pop)
        if use_ranks:
            return 1 - self.ranks(self.pop) / max(n - 1, 1)
        k = self.min_keys(self.pop)
        best, worst = k.min(), k.max()
        if not np.isfinite(worst - best) or worst == best:
            return np.ones(n)
        return (worst - k) / (worst - best)

    def step(self):
        t, T = self.iter - 1, self._planned
        frac = max(T - t, 0) / T
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
