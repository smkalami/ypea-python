"""Genetic Algorithm (GA)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real
from ..operators import NativeOperators, check_operator

_SELECTION = {"random": "random", "rand": "random",
              "roulettewheel": "roulettewheel", "roulette": "roulettewheel", "rw": "roulettewheel",
              "tournament": "tournament"}


class GA(Algorithm):
    """Real coded Genetic Algorithm with (mu + lambda) survival, as in YPEA for MATLAB.

    Parameters
    ----------
    crossover_prob : fraction of the population size produced by crossover
    crossover_inflation : gamma of blend crossover. Each child component is
        a*x1 + (1 - a)*x2 with a drawn from [-gamma, 1 + gamma] (BLX alpha). The
        default 0.5 is BLX-0.5 (Eshelman and Schaffer 1993). YPEA for MATLAB used 0.1.
    mutation_prob : fraction of the population size produced by mutation
    mutation_rate : fraction of components changed in a mutant. 0 turns mutation off, also in the native space
    mutation_step_size : standard deviation of the Gaussian mutation (in code units)
    mutation_step_size_damp : the step size is multiplied by this after every iteration
    selection : "roulettewheel" (rank based), "tournament" or "random"
    selection_pressure : pressure for roulette wheel selection
    tournament_size : number of competitors in tournament selection
    space : "encoded" (the default) works on the [0, 1] codes. "native" decodes each
        variable, applies its native operators (``ypea.operators``), and encodes the
        children back. Types without native operators use the code operators.
    operators : dict, optional
        Maps a variable name to an object with ``crossover(x1, x2, rng)`` and/or
        ``mutate(x, rng)`` methods, or a tuple ``(crossover, mutate)``, used in place of
        the defaults with ``space="native"``. A missing or None entry keeps the default.
    crossover : ypea.operators.Crossover or None
        An operator in place of blend crossover, for example ``ypea.operators.SBX()``.
        A code operator needs ``space="encoded"`` and works on the whole codes. A native
        one (``NativeCrossover``) needs ``space="native"`` and replaces the default of the
        variable types it lists (``operators`` entries still come first). None (the
        default) keeps blend crossover, or the native defaults.
    mutation : ypea.operators.Mutation or None
        An operator in place of the Gaussian mutation, for example
        ``ypea.operators.PolynomialMutation()``, with the same rules as ``crossover``.
        ``mutation_prob`` still sets the number of mutants, and ``mutation_rate=0`` still
        turns mutation off.

    Parents, mutants and children are merged, and the best ``pop_size`` survive.
    Roulette wheel selection uses ranks, so it does not depend on the scale or sign of
    the objective (MATLAB used raw objective values).

    References: Goldberg (1989). Eshelman and Schaffer (1993) for blend crossover.
    """

    name = "Genetic Algorithm"
    short_name = "GA"

    def __init__(self, *, crossover_prob=0.7, crossover_inflation=0.5, mutation_prob=0.3,
                 mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99,
                 selection="roulettewheel", selection_pressure=5.0, tournament_size=3, space="encoded",
                 operators=None, crossover=None, mutation=None, **kwargs):
        self.crossover_prob = crossover_prob
        self.crossover_inflation = crossover_inflation
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate
        self.mutation_step_size = mutation_step_size
        self.mutation_step_size_damp = mutation_step_size_damp
        self.selection = _SELECTION.get(str(selection).lower(), selection)
        self.selection_pressure = selection_pressure
        self.tournament_size = tournament_size
        self.space = space
        self.operators = operators
        self.crossover = crossover
        self.mutation = mutation
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
        check(is_int(self.tournament_size, 1),
              "tournament_size must be a positive integer")
        check(self.space in ("encoded", "native"), "space must be 'encoded' or 'native'")
        check(self.operators is None or isinstance(self.operators, dict),
              "operators must be a dict of variable name to operators, or None")
        check_operator(self.crossover, "crossover", self.space)
        check_operator(self.mutation, "mutation", self.space)

    def check_problem(self, problem):
        check_operator(self.crossover, "crossover", self.space, problem)
        check_operator(self.mutation, "mutation", self.space, problem)

    def initialize(self):
        self.pop = self.sort(self.random_population())
        self._sigma = self.mutation_step_size
        self._native = None
        if self.space == "native":
            self._native = NativeOperators(self.problem, self.operators, crossover=self.crossover,
                                           mutation=self.mutation, alpha=self.crossover_inflation,
                                           rate=self.mutation_rate, sigma=self.mutation_step_size)

    def select(self, count):
        """Indices of ``count`` parents from the current (sorted) population."""
        n = len(self.pop)
        if self.selection == "random":
            return self.rng.integers(n, size=count)
        if self.selection == "roulettewheel":
            return self.rng.choice(n, size=count, p=self.selection_probs(self.pop, self.selection_pressure))
        # tournament: the population is sorted, so the smallest index wins
        return self.rng.integers(n, size=(count, self.tournament_size)).min(axis=1)

    def blend_crossover(self, x1, x2):
        """Blend crossover (BLX alpha) of code batches, the default."""
        g = self.crossover_inflation
        a = self.rng.uniform(-g, 1 + g, x1.shape)
        return a * x1 + (1 - a) * x2, a * x2 + (1 - a) * x1

    def mutate(self, x):
        """Gaussian mutation of code batches, the default."""
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
            x1, x2 = self.pop.X[p[0::2]], self.pop.X[p[1::2]]
            if self._native:
                y1, y2 = self._native.apply("crossover", [x1, x2], self.rng, self.blend_crossover)
            elif self.crossover is not None:
                y1, y2 = self.crossover(x1, x2, self.rng)
            else:
                y1, y2 = self.blend_crossover(x1, x2)
            children.append(self.evaluate(np.vstack([y1, y2])))
        if nm and self.mutation_rate > 0:  # rate 0 would only make clones
            x = self.pop.X[self.rng.integers(len(self.pop), size=nm)]
            if self._native:
                self._native.set_sigma(self._sigma)
                y = self._native.apply("mutate", [x], self.rng, lambda c: (self.mutate(c),))[0]
            elif self.mutation is not None:
                y = self.mutation(x, self.rng)
            else:
                y = self.mutate(x)
            children.append(self.evaluate(y))
        merged = self.pop.concat(self.pop, *children)
        self.pop = self.sort(merged)[:n] if len(merged) > n else self.sort(merged)
        self._sigma *= self.mutation_step_size_damp
