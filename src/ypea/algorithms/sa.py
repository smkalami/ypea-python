"""Simulated Annealing (SA)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real


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
        to final_temp over max_iter iterations
    move_count : moves per solution in each sub iteration
    mutation_rate : probability that a component changes in a move (at least one changes)
    mutation_step_size : standard deviation of the Gaussian move (in code units)
    mutation_step_size_damp : the step size is multiplied by this after every iteration

    When the problem has constraints and either solution is infeasible, delta is the
    difference in constraint violation instead of objective value. The temperature is
    in the units of the objective, so set ``initial_temp`` to match its scale.

    Reference: Kirkpatrick, Gelatt and Vecchi (1983).
    """

    name = "Simulated Annealing"
    short_name = "SA"

    def __init__(self, *, max_sub_iter=10, initial_temp=1000.0, final_temp=1.0, move_count=5,
                 mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99,
                 pop_size=10, **kwargs):
        self.max_sub_iter = max_sub_iter
        self.initial_temp, self.final_temp = initial_temp, final_temp
        self.move_count = move_count
        self.mutation_rate = mutation_rate
        self.mutation_step_size = mutation_step_size
        self.mutation_step_size_damp = mutation_step_size_damp
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

    def initialize(self):
        self.pop = self.random_population()
        self._temp = self.initial_temp
        # reach final_temp at the end of the run: after max_iter iterations, or else after
        # the number of iterations that max_nfe allows
        per_iter = self.pop_size * self.move_count * self.max_sub_iter
        steps = self.max_iter or (int(np.ceil(self.max_nfe / per_iter)) if self.max_nfe else 100)
        self._temp_damp = (self.final_temp / self.initial_temp) ** (1 / steps)
        self._sigma = self.mutation_step_size

    def mutate(self, X):
        n, d = X.shape
        flag = self.rng.random((n, d)) <= self.mutation_rate
        flag[np.arange(n), self.rng.integers(d, size=n)] = True
        return X + flag * self._sigma * self.rng.standard_normal((n, d))

    def iterate(self):
        n, m = len(self.pop), self.move_count
        cmp = self.problem.comparator
        for _ in range(self.max_sub_iter):
            moves = self.evaluate(self.mutate(np.repeat(self.pop.X, m, axis=0)))
            # best move of each solution (moves of solution i are rows i*m .. i*m + m - 1)
            ranks = self.ranks(moves).reshape(n, m)
            cand = moves[np.arange(n) * m + np.argmin(ranks, axis=1)]
            accept = self.better(cand, self.pop)
            feasible = (cand.V <= 0) & (self.pop.V <= 0)
            delta = np.where(feasible,
                             cmp.sign * (cand.F[:, 0] - self.pop.F[:, 0]),
                             cand.V - self.pop.V)
            with np.errstate(invalid="ignore", over="ignore"):
                p = np.exp(-np.nan_to_num(delta, nan=np.inf, posinf=np.inf) / self._temp)
            accept |= self.rng.random(n) < p
            self.pop.replace(accept, cand)
        self._temp *= self._temp_damp
        self._sigma *= self.mutation_step_size_damp
