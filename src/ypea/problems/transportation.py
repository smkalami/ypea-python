"""Transportation problem.

Ship goods from m sources with ``supply[i]`` to n destinations with ``demand[j]`` at unit cost
``cost[i, j]``, and minimize the total cost. Each source ships all of its supply, and each
destination receives at least its demand.

The variable is ``FixedSum("x", (m, n), target=supply, axis=1)``: the decoded shipments x[i, j]
are non negative and each row sums to the supply, so the supply constraints always hold. The
demand constraints ``demand[j] - sum_i x[i, j] <= 0`` are the problem's constraints. When the
total supply equals the total demand, these hold only with equality, which a metaheuristic
reaches only approximately, so give some spare supply for search runs. The problem is vectorized.
"""

from __future__ import annotations

import numpy as np

from ..problem import Problem
from ..variables import FixedSum

__all__ = ["ShippingCost", "transportation"]


class ShippingCost:
    """Objective: total shipping cost of ``sol.x``, with one demand constraint per destination."""

    def __init__(self, cost, demand):
        self.cost = np.asarray(cost, dtype=float)
        self.demand = np.asarray(demand, dtype=float)

    def __call__(self, sol):
        x = np.asarray(sol.x, dtype=float)
        return (x * self.cost).sum(axis=(-2, -1)), self.demand - x.sum(axis=-2)


def transportation(cost, supply, demand, *, name="transportation", optimum=None, source=None):
    """A transportation problem.

    Parameters
    ----------
    cost : array of shape (m, n)
        Unit shipping cost from source i to destination j.
    supply : array of shape (m,)
    demand : array of shape (n,)
        The total supply must be at least the total demand.
    """
    cost = np.asarray(cost, dtype=float)
    supply, demand = np.asarray(supply, dtype=float), np.asarray(demand, dtype=float)
    if cost.ndim != 2 or cost.shape != (len(supply), len(demand)):
        raise ValueError("cost must have shape (len(supply), len(demand))")
    if np.any(supply < 0) or np.any(demand < 0):
        raise ValueError("supply and demand must be non negative")
    if supply.sum() < demand.sum() * (1 - 1e-12):
        raise ValueError("the total supply is below the total demand, so no solution is feasible")
    return Problem(FixedSum("x", cost.shape, target=supply, axis=1), ShippingCost(cost, demand),
                   vectorized=True, name=name, optimum=optimum,
                   info={"source": source, "supply": supply, "demand": demand})

