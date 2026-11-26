"""Knapsack problems: 0/1, multidimensional and generalized integer.

Each problem maximizes the total value ``x @ values`` subject to ``weights @ x <= capacity``.
The constraints are returned as ``g = weights @ x - capacity``, satisfied when ``g <= 0``.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np

from ..problem import Problem
from ..variables import Binary, Integer

__all__ = ["KnapsackObjective", "knapsack", "integer_knapsack", "matlab_knapsack", "mknap1"]


class KnapsackObjective:
    """Objective: total value of ``sol.x``, with one constraint per row of ``weights``.

    Works on one solution with shape (n,) and on a batch with shape (m, n).
    """

    def __init__(self, values, weights, capacity):
        self.values = np.asarray(values, dtype=float)
        self.weights = np.atleast_2d(np.asarray(weights, dtype=float))
        self.capacity = np.atleast_1d(np.asarray(capacity, dtype=float))
        n, m = len(self.values), len(self.capacity)
        if self.values.ndim != 1 or self.weights.shape != (m, n):
            raise ValueError(f"expected {n} values, weights of shape ({m}, {n}) and {m} capacities, "
                             f"got weights of shape {self.weights.shape}")

    def __call__(self, sol):
        x = np.asarray(sol.x, dtype=float)
        return x @ self.values, x @ self.weights.T - self.capacity


def knapsack(values, weights, capacity, *, name="knapsack", optimum=None, source=None):
    """A 0/1 knapsack problem, or a multidimensional one when ``weights`` is a matrix.

    Parameters
    ----------
    values : array of shape (n,)
    weights : array of shape (n,), or (m, n) for m constraints
    capacity : float, or array of shape (m,)
    optimum : float, optional
        Known optimal value.

    The variable is ``Binary("x", n)``, the sense is "max", and the problem is vectorized.
    """
    obj = KnapsackObjective(values, weights, capacity)
    return Problem(Binary("x", len(obj.values)), obj, "max", vectorized=True, name=name,
                   optimum=optimum, info={"source": source, "best_known": optimum})


def integer_knapsack(values, weights, capacity, upper, *, name="integer_knapsack", optimum=None, source=None):
    """A generalized (bounded) knapsack: item ``i`` may be packed from 0 to ``upper[i]`` times.

    The variable is ``Integer("x", n, lb=0, ub=upper)``. Otherwise as ``knapsack``.
    """
    obj = KnapsackObjective(values, weights, capacity)
    n = len(obj.values)
    upper = np.broadcast_to(np.asarray(upper), (n,))
    return Problem(Integer("x", n, lb=0, ub=upper), obj, "max", vectorized=True, name=name,
                   optimum=optimum, info={"source": source, "best_known": optimum})


def matlab_knapsack():
    """The 20 item binary knapsack of the YPEA for MATLAB documentation (capacity 20, optimum 73)."""
    v = [9, 6, 8, 7, 6, 3, 6, 1, 6, 6, 7, 9, 9, 7, 6, 9, 6, 1, 2, 8]
    w = [5, 8, 2, 5, 6, 1, 6, 4, 1, 5, 2, 2, 2, 2, 2, 1, 6, 3, 5, 7]
    x_opt = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    p = knapsack(v, w, 20, name="matlab_knapsack", optimum=73, source="YPEA for MATLAB, doc_ypea_problem")
    p.info["x_opt"] = x_opt
    return p


def mknap1():
    """The 7 multidimensional knapsack problems of OR-Library file mknap1 (Petersen 1967), bundled."""
    from .io import read_orlib_mknap

    return read_orlib_mknap(files(__package__) / "data" / "mknap1.txt")
