"""The n-Queens problem.

The queen of row ``i`` stands in column ``q[i]``. A permutation already keeps rows and
columns distinct, so the objective counts the pairs of queens that share a diagonal.
The optimum is 0 for every n except 2 and 3, where it is 1.
"""

from __future__ import annotations

import numpy as np

from ..problem import Problem
from ..variables import Permutation

__all__ = ["DiagonalAttacks", "nqueens"]


class DiagonalAttacks:
    """Objective: number of queen pairs on a common diagonal, for one (n,) or a batch (m, n)."""

    def __call__(self, sol):
        q = np.asarray(sol.q)
        n = q.shape[-1]
        rows = np.arange(n)
        dq = np.abs(q[..., :, None] - q[..., None, :])
        attack = dq == np.abs(rows[:, None] - rows[None, :])
        return np.triu(attack, k=1).sum(axis=(-2, -1))


def nqueens(n=8):
    """An n-Queens problem with the variable ``Permutation("q", n)``, vectorized."""
    n = int(n)
    optimum = 1 if n in (2, 3) else 0
    return Problem(Permutation("q", n), DiagonalAttacks(), "min", vectorized=True, name=f"queens{n}",
                   optimum=optimum, info={"source": "classic", "best_known": optimum})
