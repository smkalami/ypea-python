"""Test problems: engineering design (constrained), combinatorial (TSP, knapsack, QAP,
n-Queens), allocation and assignment (CVRP, GAP, hub location, transportation), and readers
for standard instance files."""

from .engineering import gear_train, pressure_vessel, speed_reducer, spring, three_bar_truss, welded_beam
from .gap import ORLIB_GAP_OPTIMA, gap, orlib_gap, read_orlib_gap
from .hub import cab, hub_allocation, hub_location, read_orlib_cab
from .io import (
    read_orlib_mknap,
    read_qaplib,
    read_qaplib_solution,
    read_tsplib,
    read_tsplib_tour,
    tsplib_distances,
)
from .knapsack import integer_knapsack, knapsack, matlab_knapsack, mknap1
from .nqueens import nqueens
from .qap import QAPLIB_OPTIMA, qap
from .transportation import transportation
from .tsp import TSPLIB_OPTIMA, circle_tsp, plot_tour, random_tsp, tsp
from .vrp import cvrp, cvrp_routes, plot_routes, read_cvrplib

__all__ = [
    "ORLIB_GAP_OPTIMA", "cab", "cvrp", "cvrp_routes", "gap", "gear_train", "hub_allocation", "hub_location",
    "orlib_gap", "plot_routes", "pressure_vessel", "read_cvrplib", "read_orlib_cab", "read_orlib_gap",
    "speed_reducer", "spring", "three_bar_truss", "transportation", "welded_beam",
    "QAPLIB_OPTIMA",
    "TSPLIB_OPTIMA",
    "circle_tsp",
    "integer_knapsack",
    "knapsack",
    "matlab_knapsack",
    "mknap1",
    "nqueens",
    "plot_tour",
    "qap",
    "random_tsp",
    "read_orlib_mknap",
    "read_qaplib",
    "read_qaplib_solution",
    "read_tsplib",
    "read_tsplib_tour",
    "tsp",
    "tsplib_distances",
]
