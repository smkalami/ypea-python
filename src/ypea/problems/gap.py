"""Generalized assignment problem (GAP).

Assign each of n jobs to one of m machines. Job j on machine i costs ``cost[i, j]`` and uses
``resource[i, j]`` of the machine's capacity ``capacity[i]``. The total resource used on each
machine must not exceed its capacity.

The variable is ``Integer("machine", n, lb=0, ub=m - 1)``: ``machine[j]`` is the machine of
job j. The problem is vectorized, with one capacity constraint per machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..problem import Problem
from ..variables import Integer

__all__ = ["AssignmentCost", "gap", "read_orlib_gap", "orlib_gap", "ORLIB_GAP_OPTIMA"]

#: Optimal values of the OR-Library gap1 to gap12 problems, solved as maximization problems.
ORLIB_GAP_OPTIMA = {
    "c515-1": 336, "c515-2": 327, "c515-3": 339, "c515-4": 341, "c515-5": 326,
    "c520-1": 434, "c520-2": 436, "c520-3": 420, "c520-4": 419, "c520-5": 428,
    "c525-1": 580, "c525-2": 564, "c525-3": 573, "c525-4": 570, "c525-5": 564,
    "c530-1": 656, "c530-2": 644, "c530-3": 673, "c530-4": 647, "c530-5": 664,
    "c824-1": 563, "c824-2": 558, "c824-3": 564, "c824-4": 568, "c824-5": 559,
    "c832-1": 761, "c832-2": 759, "c832-3": 758, "c832-4": 752, "c832-5": 747,
    "c840-1": 942, "c840-2": 949, "c840-3": 968, "c840-4": 945, "c840-5": 951,
    "c848-1": 1133, "c848-2": 1134, "c848-3": 1141, "c848-4": 1117, "c848-5": 1127,
    "c1030-1": 709, "c1030-2": 717, "c1030-3": 712, "c1030-4": 723, "c1030-5": 706,
    "c1040-1": 958, "c1040-2": 963, "c1040-3": 960, "c1040-4": 947, "c1040-5": 947,
    "c1050-1": 1139, "c1050-2": 1178, "c1050-3": 1195, "c1050-4": 1171, "c1050-5": 1171,
    "c1060-1": 1451, "c1060-2": 1449, "c1060-3": 1433, "c1060-4": 1447, "c1060-5": 1446,
}

_DATA = Path(__file__).parent / "data"


class AssignmentCost:
    """Objective: total cost of the assignment ``sol.machine``, with resource used - capacity per machine."""

    def __init__(self, cost, resource, capacity):
        self.cost = np.asarray(cost, dtype=float)
        self.resource = np.asarray(resource, dtype=float)
        self.capacity = np.asarray(capacity, dtype=float)

    def __call__(self, sol):
        x = np.asarray(sol.machine)
        jobs = np.arange(self.cost.shape[1])
        f = self.cost[x, jobs].sum(axis=-1)
        used = self.resource[x, jobs]
        onehot = x[..., None, :] == np.arange(len(self.capacity))[:, None]
        g = (onehot * used[..., None, :]).sum(axis=-1) - self.capacity
        return f, g


def gap(cost, resource, capacity, *, sense="min", name="gap", optimum=None, source=None):
    """A generalized assignment problem.

    Parameters
    ----------
    cost : array of shape (m, n)
        Cost (or profit, with ``sense="max"``) of each job on each machine.
    resource : array of shape (m, n)
        Capacity used by each job on each machine.
    capacity : array of shape (m,)
    sense : "min" or "max"
    name, optimum, source : optional metadata.
    """
    cost, resource = np.asarray(cost, dtype=float), np.asarray(resource, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    if cost.ndim != 2 or resource.shape != cost.shape or capacity.shape != (cost.shape[0],):
        raise ValueError("cost and resource must have shape (m, n) and capacity shape (m,)")
    m, n = cost.shape
    return Problem(Integer("machine", n, lb=0, ub=m - 1), AssignmentCost(cost, resource, capacity),
                   sense=sense, vectorized=True, name=name, optimum=optimum, info={"source": source})


def read_orlib_gap(path, sense="max"):
    """Read an OR-Library GAP file (gap1 to gap12, gapa to gapd) and return a list of ``Problem``.

    The file holds several problems: for each, m and n, the (m, n) cost matrix, the (m, n)
    resource matrix and the m capacities. gap1 to gap12 are usually solved as maximization
    problems (the default here), and their problems get the optimum from ``ORLIB_GAP_OPTIMA``.
    The type A to D files are minimization problems, so read them with ``sense="min"``.
    Problems are named as in OR-Library, for example "c515-1" (5 machines, 15 jobs, number 1).
    """
    t = np.array(Path(path).read_text().split(), dtype=float)
    count, pos, problems = int(t[0]), 1, []
    sizes = {}
    for k in range(count):
        m, n = int(t[pos]), int(t[pos + 1])
        pos += 2
        need = 2 * m * n + m
        if len(t) < pos + need:
            raise ValueError(f"{path}: problem {k + 1} is incomplete")
        cost = t[pos:pos + m * n].reshape(m, n)
        resource = t[pos + m * n:pos + 2 * m * n].reshape(m, n)
        capacity = t[pos + 2 * m * n:pos + need]
        pos += need
        sizes[(m, n)] = sizes.get((m, n), 0) + 1
        name = f"c{m}{n}-{sizes[(m, n)]}"
        optimum = ORLIB_GAP_OPTIMA.get(name) if sense == "max" else None
        problems.append(gap(cost, resource, capacity, sense=sense, name=name, optimum=optimum,
                            source=f"OR-Library {Path(path).stem} {name}"))
    return problems


def orlib_gap(index=1):
    """Problem c515-``index`` (5 machines, 15 jobs, maximization) of the bundled OR-Library file gap1."""
    if not 1 <= index <= 5:
        raise ValueError("index must be from 1 to 5")
    return read_orlib_gap(_DATA / "gap_orlib_gap1.txt")[index - 1]
