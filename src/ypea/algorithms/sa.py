"""Simulated Annealing (SA)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real
from ..operators import NativeOperators, check_operator


class SA(Algorithm):
    """Population based Simulated Annealing, as in YPEA for MATLAB.

    Each of the ``pop_size`` current solutions makes ``move_count`` random moves per
    sub iteration and keeps the best of them as its candidate. The candidate is
    accepted if it is better, or otherwise with probability exp(-delta / T)
    (Metropolis rule). With ``pop_size=1`` and ``move_count=1`` this is classic SA.

    Parameters
    ----------
    max_sub_iter : sub iterations at a fixed temperature
    initial_temp, final_temp : the temperature falls geometrically from initial_temp
        to final_temp over the planned length of the run
    move_count : moves per solution in each sub iteration
    mutation_rate : probability that a component changes in a move (at least one changes)
    mutation_step_size : standard deviation of the Gaussian move (in code units)
    mutation_step_size_damp : the step size is multiplied by this after every iteration
    space : "encoded" (the default) moves in the [0, 1] codes. "native" decodes each
        variable, applies its native mutation (``ypea.operators``), and encodes the move
        back. Types without native operators use the Gaussian move on their codes.
    operators : dict, optional
        Maps a variable name to an object with a ``mutate(x, rng)`` method, or a tuple
        ``(None, mutate)``, used in place of the default move with ``space="native"``.
    mutation : ypea.operators.Mutation or None
        An operator in place of the Gaussian move, for example
        ``ypea.operators.PolynomialMutation()``. A code operator needs ``space="encoded"``,
        a native one (``NativeMutation``) needs ``space="native"`` and replaces the move of
        the variable types it lists. None (the default) keeps the default moves. SA does
        not recombine, so it has no crossover setting.

    delta follows the problem's comparison rules. With feasibility rules (the default) it
    is the difference in constraint violation when the violations differ, and the
    difference in objective value otherwise (so also for two feasible solutions). With
    a penalty it is the difference in penalized objective. The temperature is in those
    units, so set ``initial_temp`` to match the scale of the objective. The temperature
    never falls below ``final_temp``, even when the run lasts longer than planned.

    Reference: Kirkpatrick, Gelatt and Vecchi (1983).
    """

    name = "Simulated Annealing"
    short_name = "SA"

    def __init__(self, *, max_sub_iter=10, initial_temp=1000.0, final_temp=1.0, move_count=5,
                 mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99,
                 pop_size=10, space="encoded", operators=None, mutation=None, **kwargs):
        self.max_sub_iter = max_sub_iter
        self.initial_temp, self.final_temp = initial_temp, final_temp
        self.move_count = move_count
        self.mutation_rate = mutation_rate
        self.mutation_step_size = mutation_step_size
        self.mutation_step_size_damp = mutation_step_size_damp
        self.space = space
        self.operators = operators
        self.mutation = mutation
        super().__init__(pop_size=pop_size, **kwargs)

    def validate(self):
        super().validate()
        check(is_int(self.max_sub_iter, 1), "max_sub_iter must be a positive integer")
        check(is_int(self.move_count, 1), "move_count must be a positive integer")
        check(is_real(self.initial_temp, 0) and self.initial_temp > 0, "initial_temp must be positive")
        check(is_real(self.final_temp, 0) and 0 < self.final_temp <= self.initial_temp,
              "final_temp must be positive and not above initial_temp")
        check(is_real(self.mutation_rate, 0, 1), "mutation_rate must be in [0, 1]")
        check(is_real(self.mutation_step_size, 0), "mutation_step_size must be non negative")
        check(is_real(self.mutation_step_size_damp, 0) and self.mutation_step_size_damp > 0,
              "mutation_step_size_damp must be positive")
        check(self.space in ("encoded", "native"), "space must be 'encoded' or 'native'")
        check(self.operators is None or isinstance(self.operators, dict),
              "operators must be a dict of variable name to operators, or None")
        check_operator(self.mutation, "mutation", self.space)

    def check_problem(self, problem):
        check_operator(self.mutation, "mutation", self.space, problem)

    def initialize(self):
        self.pop = self.random_population()
        self._temp = self.initial_temp
        # reach final_temp when the run ends, whichever limit binds first
        steps = self.planned_iterations(self.pop_size * self.move_count * self.max_sub_iter)
        self._temp_damp = (self.final_temp / self.initial_temp) ** (1 / steps)
        self._sigma = self.mutation_step_size
        self._native = None
        if self.space == "native":
            self._native = NativeOperators(self.problem, self.operators, mutation=self.mutation,
                                           rate=self.mutation_rate, sigma=self.mutation_step_size)

    def move(self, X):
        """Random moves of the solutions with codes ``X``."""
        if not self._native:
            return self.mutate(X) if self.mutation is None else self.mutation(X, self.rng)
        self._native.set_sigma(self._sigma)
        return self._native.apply("mutate", [X], self.rng, lambda c: (self.mutate(c),))[0]

    def mutate(self, X):
        n, d = X.shape
        flag = self.rng.random((n, d)) <= self.mutation_rate
        flag[np.arange(n), self.rng.integers(d, size=n)] = True
        return X + flag * self._sigma * self.rng.standard_normal((n, d))

    def iterate(self):
        n, m = len(self.pop), self.move_count
        cmp = self.problem.comparator
        for _ in range(self.max_sub_iter):
            moves = self.evaluate(self.move(np.repeat(self.pop.X, m, axis=0)))
            # best move of each solution (moves of solution i are rows i*m .. i*m + m - 1)
            ranks = self.ranks(moves).reshape(n, m)
            cand = moves[np.arange(n) * m + np.argmin(ranks, axis=1)]
            accept = self.better(cand, self.pop)
            # delta from the comparator keys, so "worse" means the same as everywhere else:
            # the primary key difference, or the secondary one when the primary keys are equal.
            # inf - inf gives NaN, which counts as an infinitely worse move (never accepted).
            pc, sc = cmp.keys(cand.F, cand.V)
            po, so = cmp.keys(self.pop.F, self.pop.V)
            with np.errstate(invalid="ignore", over="ignore"):
                delta = np.where(pc == po, sc - so, pc - po)
                p = np.exp(-np.nan_to_num(delta, nan=np.inf, posinf=np.inf) / self._temp)
            accept |= self.rng.random(n) < p
            self.pop.replace(accept, cand)
        self._temp = max(self._temp * self._temp_damp, self.final_temp)
        self._sigma *= self.mutation_step_size_damp
