"""Uncapacitated single allocation p-hub median problem.

Choose p of the n nodes as hubs and allocate every node to one hub (a hub to itself). The flow
W[i, j] from node i to node j travels i -> hub(i) -> hub(j) -> j, and its cost per unit is
chi C[i, hub(i)] + alpha C[hub(i), hub(j)] + delta C[hub(j), j]. The objective is the total
cost over all pairs. The discount factor alpha (< 1) models the cheaper transport between hubs.

Variables: ``Selection("hubs", n, p)`` gives the hub nodes, and ``Integer("alloc", n, 0, p - 1)``
gives each node's hub as an index into ``hubs``. The allocation of a hub node is ignored, since
a hub serves itself. The problem is vectorized.

The CAB data set (25 US cities, O'Kelly 1987) from OR-Library is bundled, see ``cab``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..problem import Problem
from ..variables import Integer, Selection

__all__ = ["HubCost", "hub_location", "read_orlib_cab", "cab", "hub_allocation"]

_DATA = Path(__file__).parent / "data"


def hub_allocation(hubs, alloc):
    """The hub node of every node: ``hubs[alloc]``, with each hub allocated to itself."""
    hubs, alloc = np.asarray(hubs), np.asarray(alloc)
    node_hub = np.take_along_axis(hubs, alloc, axis=-1)
    n = alloc.shape[-1]
    is_hub = (np.arange(n) == hubs[..., :, None]).any(axis=-2)
    return np.where(is_hub, np.arange(n), node_hub)


class HubCost:
    """Objective: total transport cost of a single allocation hub network."""

    def __init__(self, flows, costs, alpha, chi=1.0, delta=1.0):
        self.flows = np.asarray(flows, dtype=float)
        self.costs = np.asarray(costs, dtype=float)
        self.alpha, self.chi, self.delta = float(alpha), float(chi), float(delta)

    def __call__(self, sol):
        a = np.atleast_2d(hub_allocation(sol.hubs, sol.alloc))
        nodes = np.arange(a.shape[1])
        out_flow, in_flow = self.flows.sum(axis=1), self.flows.sum(axis=0)
        access = self.costs[nodes, a]  # C[i, hub(i)]
        f = (self.chi * access @ out_flow + self.delta * self.costs[a, nodes] @ in_flow
             + self.alpha * (self.costs[a[:, :, None], a[:, None, :]] * self.flows).sum(axis=(1, 2)))
        return f if np.ndim(sol.alloc) > 1 else f[0]


def hub_location(flows, costs, p, alpha, *, chi=1.0, delta=1.0, name="hub_location", optimum=None,
                 source=None):
    """A single allocation p-hub median problem.

    Parameters
    ----------
    flows : array of shape (n, n)
        Flow W[i, j] from node i to node j.
    costs : array of shape (n, n)
        Unit transport cost (usually the distance) C[i, j].
    p : int
        Number of hubs.
    alpha : float
        Discount factor for transport between hubs.
    chi, delta : float
        Factors for collection (node to hub) and distribution (hub to node), 1 by default.
    """
    flows, costs = np.asarray(flows, dtype=float), np.asarray(costs, dtype=float)
    n = len(flows)
    if flows.shape != (n, n) or costs.shape != (n, n):
        raise ValueError("flows and costs must be square matrices of the same size")
    if not 1 <= p <= n:
        raise ValueError("p must be between 1 and n")
    return Problem([Selection("hubs", n, p), Integer("alloc", n, lb=0, ub=p - 1)],
                   HubCost(flows, costs, alpha, chi, delta), vectorized=True, name=name, optimum=optimum,
                   info={"source": source, "p": p, "alpha": alpha})


def read_orlib_cab(path):
    """Read the OR-Library CAB file (phub4). Returns ``(flows, costs)``, both 25 by 25.

    The file lists ``i j W[i, j] C[i, j]`` for every pair, after the line starting with "param".
    """
    lines = Path(path).read_text().splitlines()
    start = next(k for k, line in enumerate(lines) if line.lstrip().startswith("param"))
    rows = [line.replace(";", " ").split() for line in lines[start + 1:]]
    data = np.array([[float(v) for v in r[:4]] for r in rows if len(r) >= 4])
    n = int(data[:, 0].max())
    flows, costs = np.zeros((n, n)), np.zeros((n, n))
    i, j = data[:, 0].astype(int) - 1, data[:, 1].astype(int) - 1
    flows[i, j], costs[i, j] = data[:, 2], data[:, 3]
    return flows, costs


def cab(n=25, p=2, alpha=0.2, *, normalize=True):
    """CAB p-hub median instance (O'Kelly 1987), bundled from OR-Library.

    ``n`` is 10, 15, 20 or 25: the subsets are the first n cities. With ``normalize`` (default,
    as in most papers), the flows are divided by their total, so the objective is the mean cost
    per unit of flow. Collection and distribution factors are 1.

    Reference: M. E. O'Kelly, "A quadratic integer program for the location of interacting hub
    facilities", European Journal of Operational Research 32 (1987) 393-404.
    """
    if n not in (10, 15, 20, 25):
        raise ValueError("n must be 10, 15, 20 or 25")
    flows, costs = read_orlib_cab(_DATA / "hub_orlib_phub4.txt")
    flows, costs = flows[:n, :n], costs[:n, :n]
    if normalize:
        flows = flows / flows.sum()
    return hub_location(flows, costs, p, alpha, name=f"cab{n}_p{p}_a{alpha:g}", source="OR-Library phub4 (CAB)")
