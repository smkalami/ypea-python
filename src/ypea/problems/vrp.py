"""Capacitated vehicle routing problem (CVRP).

Node 0 is the depot and nodes 1 to n are the customers. Each of at most k vehicles leaves the
depot, visits some customers and comes back. Every customer is visited once, and the demand
served on one route must not exceed the vehicle capacity.

Encoding: one ``Permutation("route", n + k - 1)``. Values 0 to n - 1 are the customers (node
value + 1), and values n and above are separators between routes. Reading the permutation from
left to right gives up to k routes in their visiting order, and a route may be empty (fewer
vehicles used). This is the same code as ``Partition(n, k)`` with the order kept inside each
group, but it decodes to an array, so the problem is vectorized.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..problem import Problem
from ..variables import Permutation
from .io import _explicit_matrix, _parse_tsplib, tsplib_distances

__all__ = ["RouteCost", "cvrp", "cvrp_routes", "read_cvrplib", "plot_routes"]


class RouteCost:
    """Objective: total route length, with one capacity constraint per vehicle (load - capacity)."""

    def __init__(self, distances, demands, capacity, n_vehicles):
        self.distances = np.asarray(distances, dtype=float)
        self.demands = np.asarray(demands, dtype=float)
        self.capacity = float(capacity)
        self.n_vehicles = int(n_vehicles)

    def __call__(self, sol):
        perm = np.atleast_2d(sol.route)
        n = len(self.demands) - 1
        nodes = np.where(perm < n, perm + 1, 0)  # separators go back to the depot
        depot = np.zeros((len(perm), 1), dtype=nodes.dtype)
        tour = np.hstack([depot, nodes, depot])
        f = self.distances[tour[:, :-1], tour[:, 1:]].sum(axis=1)
        vehicle = np.cumsum(perm >= n, axis=1)
        load = np.zeros((len(perm), self.n_vehicles))
        np.add.at(load, (np.arange(len(perm))[:, None], vehicle), self.demands[nodes])
        g = load - self.capacity
        if np.ndim(sol.route) == 1:
            return f[0], g[0]
        return f, g


def cvrp(coords=None, demands=None, capacity=None, n_vehicles=None, *, distances=None, name="cvrp",
         optimum=None, source=None):
    """A CVRP from node coordinates or a distance matrix.

    Parameters
    ----------
    coords : array of shape (n + 1, 2), optional
        Node coordinates, the depot first. Without ``distances``, distances are Euclidean.
    demands : array of shape (n + 1,)
        Demand of each node. The depot's demand (the first value) must be 0.
    capacity : float
        Vehicle capacity.
    n_vehicles : int
        Number of vehicles k (the most routes a solution may have).
    distances : array of shape (n + 1, n + 1), optional
    name, optimum, source : optional metadata. The coordinates go to ``info["coords"]``.

    The variable is ``Permutation("route", n + k - 1)``, the objective is the total distance,
    and there are k capacity constraints. Use ``cvrp_routes`` to get the routes of a solution.
    """
    if coords is None and distances is None:
        raise ValueError("give coords or distances")
    demands = np.asarray(demands, dtype=float)
    if demands.ndim != 1 or len(demands) < 2 or demands[0] != 0:
        raise ValueError("demands must be a vector with the depot first and demand 0 at the depot")
    if np.any(demands < 0):
        raise ValueError("demands must be non negative")
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (len(demands), 2):
            raise ValueError(f"coords must have shape ({len(demands)}, 2)")
    if distances is None:
        distances = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    distances = np.asarray(distances, dtype=float)
    if distances.shape != (len(demands), len(demands)):
        raise ValueError("distances must be a square matrix with one row per node")
    n_vehicles = int(n_vehicles)
    if n_vehicles < 1 or capacity is None or capacity <= 0:
        raise ValueError("n_vehicles must be at least 1 and capacity positive")
    n = len(demands) - 1
    return Problem(Permutation("route", n + n_vehicles - 1), RouteCost(distances, demands, capacity, n_vehicles),
                   vectorized=True, name=name, optimum=optimum,
                   info={"coords": coords, "source": source, "n_customers": n, "n_vehicles": n_vehicles})


def cvrp_routes(problem, solution):
    """The routes of a solution: a list of k arrays of node indices (customers are 1 to n).

    Empty routes are kept, so route i is vehicle i. ``solution`` is a ``Solution`` or the
    permutation itself.
    """
    perm = np.asarray(getattr(solution, "route", solution))
    n = problem.info["n_customers"]
    cuts = np.flatnonzero(perm >= n)
    return [r[r < n] + 1 for r in np.split(perm, cuts)]


def read_cvrplib(path, n_vehicles=None, *, rounding=True, optimum=None):
    """Read a CVRPLIB file (TSPLIB like ``.vrp`` format) and return the CVRP ``Problem``.

    Needs NODE_COORD_SECTION (EUC_2D) or an EXPLICIT edge weight section, DEMAND_SECTION and
    DEPOT_SECTION with one depot. ``n_vehicles`` defaults to the ``-kN`` part of the name (for
    example A-n32-k5), else to ceil(total demand / capacity). With ``rounding`` (default, the
    CVRPLIB convention for EUC_2D), distances are rounded to the nearest integer. ``optimum``
    defaults to a value given in the COMMENT line ("Optimal value: 784"). In some sets the
    ``-kN`` number is the fewest vehicles possible, and the best known solution uses more
    routes, so pass a larger ``n_vehicles`` when the instance needs it.
    """
    spec, sections = _parse_tsplib(Path(path).read_text())
    n = int(spec["DIMENSION"])
    kind = spec.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    coords = None
    if "NODE_COORD_SECTION" in sections:
        coords = np.asarray(sections["NODE_COORD_SECTION"]).reshape(n, -1)[:, 1:3]
    if kind == "EUC_2D":
        d = tsplib_distances(coords, kind) if rounding else \
            np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    elif kind == "EXPLICIT":
        d = _explicit_matrix(sections["EDGE_WEIGHT_SECTION"], n, spec.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX"))
    else:
        raise ValueError(f"unsupported edge weight type {kind!r}, expected EUC_2D or EXPLICIT")
    demand = np.asarray(sections["DEMAND_SECTION"]).reshape(n, 2)[:, 1]
    depots = [int(v) - 1 for v in sections.get("DEPOT_SECTION", [1]) if v >= 1]
    if len(depots) != 1:
        raise ValueError("exactly one depot is supported")
    order = [depots[0]] + [i for i in range(n) if i != depots[0]]
    capacity = float(spec["CAPACITY"])
    name = spec.get("NAME", Path(path).stem)
    if n_vehicles is None:
        m = re.search(r"-k(\d+)", name)
        n_vehicles = int(m.group(1)) if m else int(np.ceil(demand.sum() / capacity))
    if optimum is None:
        m = re.search(r"optimal value:\s*([\d.]+)", spec.get("COMMENT", ""), re.IGNORECASE)
        optimum = float(m.group(1)) if m else None
    return cvrp(None if coords is None else coords[order], demand[order], capacity, n_vehicles,
                distances=d[np.ix_(order, order)], name=name, optimum=optimum, source=f"CVRPLIB {name}")


def plot_routes(problem, solution, ax=None):
    """Plot the routes of a CVRP solution (needs coordinates and matplotlib). Returns the axes."""
    import matplotlib.pyplot as plt

    coords = problem.info.get("coords")
    if coords is None:
        raise ValueError("the problem has no coordinates to plot")
    if ax is None:
        ax = plt.gca()
    for i, r in enumerate(cvrp_routes(problem, solution)):
        if len(r):
            path = coords[np.concatenate([[0], r, [0]])]
            ax.plot(path[:, 0], path[:, 1], "-o", ms=4, label=f"route {i + 1}")
    ax.plot(*coords[0], "ks", ms=9, label="depot")
    ax.set_aspect("equal", adjustable="datalim")
    return ax
