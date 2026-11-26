"""Traveling salesman problem (TSP).

A tour is a permutation of the cities (0 based). Its length is the sum of the distances
between consecutive cities, including the edge back from the last city to the first.
"""

from __future__ import annotations

import numpy as np

from ..problem import Problem, Solution
from ..variables import Permutation

__all__ = ["TourLength", "tsp", "random_tsp", "circle_tsp", "plot_tour", "TSPLIB_OPTIMA"]

#: Optimal tour lengths from TSPLIB (``TSP-BEST.html``), by instance name.
TSPLIB_OPTIMA = {
    "att48": 10628, "bayg29": 1610, "bays29": 2020, "berlin52": 7542, "burma14": 3323,
    "eil51": 426, "eil76": 538, "gr17": 2085, "gr21": 2707, "gr24": 1272,
    "kroA100": 21282, "pr76": 108159, "st70": 675, "ulysses16": 6859, "ulysses22": 7013,
}


class TourLength:
    """Objective: length of the closed tour ``sol.tour`` on the distance matrix ``distances``.

    Works on one tour with shape (n,) and on a batch with shape (m, n).
    """

    def __init__(self, distances):
        self.distances = np.asarray(distances, dtype=float)

    def __call__(self, sol):
        tour = np.asarray(sol.tour)
        return self.distances[tour, np.roll(tour, -1, axis=-1)].sum(axis=-1)


def tsp(coords=None, distances=None, *, name="tsp", optimum=None, source=None):
    """A TSP from city coordinates or from a distance matrix.

    Parameters
    ----------
    coords : array of shape (n, 2), optional
        City coordinates. Without ``distances``, the distances are Euclidean.
        With ``distances``, they are kept only for plotting.
    distances : array of shape (n, n), optional
        Distance matrix. It may be asymmetric: the tour goes from ``tour[i]`` to ``tour[i + 1]``.
    name : str
    optimum : float, optional
        Known optimal tour length.
    source : str, optional
        Where the instance comes from, stored in ``info["source"]``.

    The variable is ``Permutation("tour", n)`` and the problem is vectorized.
    """
    if coords is None and distances is None:
        raise ValueError("give coords or distances")
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
    if distances is None:
        distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    distances = np.asarray(distances, dtype=float)
    n = len(distances)
    if distances.shape != (n, n) or n < 2:
        raise ValueError(f"distances must be a square matrix with n >= 2, got {distances.shape}")
    if coords is not None and len(coords) != n:
        raise ValueError("coords and distances have different numbers of cities")
    info = {"source": source, "best_known": optimum, "coords": coords}
    return Problem(Permutation("tour", n), TourLength(distances), "min", vectorized=True,
                   name=name, optimum=optimum, info=info)


def random_tsp(n, seed=None, *, size=100.0):
    """A TSP with ``n`` cities placed uniformly at random in a square of side ``size``."""
    rng = np.random.default_rng(seed)
    return tsp(rng.uniform(0, size, (int(n), 2)), name=f"random_tsp{n}", source="random")


def circle_tsp(n=30, radius=40.0, seed=None, *, center=(50.0, 50.0)):
    """The TSP of the YPEA for MATLAB SA example: ``n`` cities on a circle, in shuffled order.

    The optimal tour follows the circle, so the optimum is the perimeter of the regular
    polygon, ``2 n radius sin(pi / n)``. (The MATLAB example used ``2 pi radius``, the
    circumference, which no tour reaches.)
    """
    rng = np.random.default_rng(seed)
    angle = 2 * np.pi * rng.permutation(int(n)) / n
    coords = np.column_stack([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])
    return tsp(coords, name=f"circle_tsp{n}", optimum=2 * n * radius * np.sin(np.pi / n),
               source="YPEA for MATLAB, SA example")


def plot_tour(problem, tour, ax=None, **kwargs):
    """Plot a closed tour on the city coordinates of ``problem`` (needs matplotlib).

    ``tour`` is a permutation, a ``Solution`` with a ``tour`` attribute, or a ``Result``.
    Returns the axes.
    """
    import matplotlib.pyplot as plt

    coords = problem.info.get("coords")
    if coords is None:
        raise ValueError("this problem has no city coordinates to plot")
    if hasattr(tour, "best"):
        tour = tour.best.solution
    tour = np.asarray(getattr(tour, "tour", tour))
    if ax is None:
        ax = plt.figure().gca()
    closed = np.append(tour, tour[0])
    ax.plot(coords[closed, 0], coords[closed, 1], "-", **kwargs)
    ax.plot(coords[:, 0], coords[:, 1], "o", color="black", markersize=4)
    ax.set_aspect("equal")
    ax.set_title(f"{problem.name}: length {problem.objective(Solution(tour=tour)):g}")
    return ax
