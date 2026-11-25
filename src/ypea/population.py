"""Population (arrays) and Individual (a readable view of one solution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class Population:
    """A set of solutions stored as arrays.

    ``X`` holds the codes in [0, 1] with shape (n, d), ``F`` the objective values with
    shape (n, m) (m = 1 for now), and ``V`` the total constraint violation with shape (n,).
    """

    __slots__ = ("X", "F", "V")

    def __init__(self, X, F, V):
        self.X = np.asarray(X, dtype=float)
        self.F = np.asarray(F, dtype=float).reshape(len(self.X), -1)
        self.V = np.asarray(V, dtype=float).reshape(len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if np.isscalar(idx):
            idx = [idx]
        return Population(self.X[idx], self.F[idx], self.V[idx])

    def copy(self):
        return Population(self.X.copy(), self.F.copy(), self.V.copy())

    def replace(self, mask, other: Population):
        """Replace rows where ``mask`` is True with the same rows of ``other`` (in place)."""
        self.X[mask] = other.X[mask]
        self.F[mask] = other.F[mask]
        self.V[mask] = other.V[mask]

    @staticmethod
    def concat(*pops: Population) -> Population:
        return Population(
            np.concatenate([p.X for p in pops]),
            np.concatenate([p.F for p in pops]),
            np.concatenate([p.V for p in pops]),
        )

    @property
    def obj_value(self) -> np.ndarray:
        return self.F[:, 0]

    def __repr__(self):
        return f"Population(n={len(self)}, d={self.X.shape[1]})"


@dataclass
class Individual:
    """One solution: its code, decoded solution, objective value and constraint violation."""

    position: np.ndarray
    solution: Any
    obj_value: float
    violation: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.violation <= 0
