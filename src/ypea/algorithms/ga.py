"""Genetic Algorithm (GA)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_real

_SELECTION = {"random": "random", "rand": "random",
              "roulettewheel": "roulettewheel", "roulette": "roulettewheel", "rw": "roulettewheel",
              "tournament": "tournament"}


class GA(Algorithm):
    """Real coded Genetic Algorithm with (mu + lambda) survival, as in YPEA for MATLAB.

    Parameters
    ----------
    crossover_prob : fraction of the population size produced by crossover
    crossover_inflation : gamma of blend crossover. Each child component is
        a*x1 + (1 - a)*x2 with a drawn from [-gamma, 1 + gamma] (BLX alpha).
    mutation_prob : fraction of the population size produced by mutation
    mutation_rate : fraction of components changed in a mutant
    mutation_step_size : standard deviation of the Gaussian mutation (in code units)
    mutation_step_size_damp : the step size is multiplied by this after every iteration
    selection : "roulettewheel" (rank based), "tournament" or "random"
    selection_pressure : pressure for roulette wheel selection
    tournament_size : number of competitors in tournament selection

    Parents, mutants and children are merged, and the best ``pop_size`` survive.
    Roulette wheel selection uses ranks, so it does not depend on the scale or sign of
    the objective (MATLAB used raw objective values).

    References: Goldberg (1989). Eshelman and Schaffer (1993) for blend crossover.
    """

    name = "Genetic Algorithm"
    short_name = "GA"

    def __init__(self, *, crossover_prob=0.7, crossover_inflation=0.1, mutation_prob=0.3,
                 mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99,
                 selection="roulettewheel", selection_pressure=5.0, tournament_size=3, **kwargs):
        self.crossover_prob = crossover_prob
        self.crossover_inflation = crossover_inflation
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate
        self.mutation_step_size = mutation_step_size
        self.mutation_step_size_damp = mutation_step_size_damp
        self.selection = _SELECTION.get(str(selection).lower(), selection)
        self.selection_pressure = selection_pressure
        self.tournament_size = tournament_size
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_real(self.crossover_prob, 0), "crossover_prob must be non negative")
        check(is_real(self.mutation_prob, 0), "mutation_prob must be non negative")
        check(is_real(self.crossover_inflation, 0), "crossover_inflation must be non negative")
        check(is_real(self.mutation_rate, 0, 1), "mutation_rate must be in [0, 1]")
        check(is_real(self.mutation_step_size, 0), "mutation_step_size must be non negative")
        check(is_real(self.mutation_step_size_damp, 0) and self.mutation_step_size_damp > 0,
              "mutation_step_size_damp must be positive")
        check(self.selection in _SELECTION.values(),
              "selection must be 'roulettewheel', 'tournament' or 'random'")
        check(is_real(self.selection_pressure, 0), "selection_pressure must be non negative")
        check(isinstance(self.tournament_size, int) and self.tournament_size >= 1,
              "tournament_size must be a positive integer")

    def initialize(self):
        self.pop = self.sort(self.random_population())
        self._sigma = self.mutation_step_size

    def select(self, count):
        """Indices of ``count`` parents from the current (sorted) population."""
        n = len(self.pop)
        if self.selection == "random":
            return self.rng.integers(n, size=count)
        if self.selection == "roulettewheel":
            return self.rng.choice(n, size=count, p=self.selection_probs(self.pop, self.selection_pressure))
        # tournament: the population is sorted, so the smallest index wins
        return self.rng.integers(n, size=(count, self.tournament_size)).min(axis=1)

    def crossover(self, x1, x2):
        g = self.crossover_inflation
        a = self.rng.uniform(-g, 1 + g, x1.shape)
        return a * x1 + (1 - a) * x2, a * x2 + (1 - a) * x1

    def mutate(self, x):
        n, d = x.shape
        k = int(np.ceil(self.mutation_rate * d))
        cols = np.argsort(self.rng.random((n, d)), axis=1)[:, :k]
        y = x.copy()
        rows = np.arange(n)[:, None]
        y[rows, cols] += self._sigma * self.rng.standard_normal((n, k))
        return y

    def iterate(self):
        n = self.pop_size
        nc = 2 * round(self.crossover_prob * n / 2)
        nm = round(self.mutation_prob * n)
        children = []
        if nc:
            p = self.select(nc)
            y1, y2 = self.crossover(self.pop.X[p[0::2]], self.pop.X[p[1::2]])
            children.append(self.evaluate(np.vstack([y1, y2])))
        if nm:
            children.append(self.evaluate(self.mutate(self.pop.X[self.rng.integers(len(self.pop), size=nm)])))
        merged = self.pop.concat(self.pop, *children)
        self.pop = self.sort(merged)[:n] if len(merged) > n else self.sort(merged)
        self._sigma *= self.mutation_step_size_damp
