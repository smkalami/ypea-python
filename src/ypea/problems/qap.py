"""Quadratic assignment problem (QAP).

Facility ``i`` goes to location ``p[i]``. The cost is
``sum over i, j of flow[i, j] * distance[p[i], p[j]]``, as in QAPLIB.
"""

from __future__ import annotations

import numpy as np

from ..problem import Problem
from ..variables import Permutation

__all__ = ["QAPCost", "qap", "QAPLIB_OPTIMA"]

#: Proven optimal values from QAPLIB, by instance name.
QAPLIB_OPTIMA = {
    "chr12a": 9552, "esc16a": 68, "had12": 1652, "nug12": 578, "nug14": 1014,
    "nug15": 1150, "scr12": 31410, "tai12a": 224416,
}


class QAPCost:
    """Objective: QAP cost of the assignment ``sol.p``, for one (n,) or a batch (m, n)."""

    def __init__(self, flow, distance):
        self.flow = np.asarray(flow, dtype=float)
        self.distance = np.asarray(distance, dtype=float)

    def __call__(self, sol):
        p = np.asarray(sol.p)
        return (self.flow * self.distance[p[..., :, None], p[..., None, :]]).sum(axis=(-2, -1))


def qap(flow, distance, *, name="qap", optimum=None, source=None):
    """A QAP from a flow matrix and a distance matrix, both (n, n).

    The variable is ``Permutation("p", n)`` and the problem is vectorized.
    """
    obj = QAPCost(flow, distance)
    n = len(obj.flow)
    if obj.flow.shape != (n, n) or obj.distance.shape != (n, n):
        raise ValueError(f"flow and distance must be square matrices of the same size, "
                         f"got {obj.flow.shape} and {obj.distance.shape}")
    return Problem(Permutation("p", n), obj, "min", vectorized=True, name=name, optimum=optimum,
                   info={"source": source, "best_known": optimum})
