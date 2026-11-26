"""YPEA: Yarpiz Evolutionary Algorithms.

Define an optimization problem once, then solve it with any of the algorithms::

    import numpy as np
    import ypea

    problem = ypea.Problem(
        vars=ypea.Real("x", shape=20, lb=-10, ub=10),
        objective=lambda sol: np.sum(sol.x**2),
    )
    result = ypea.PSO(max_iter=100).solve(problem, seed=1)
    print(result.best.solution.x, result.best.obj_value)
"""

from . import benchmarks, callbacks, experiment, operators, plot, problems, random, tuning
from .algorithm import Algorithm
from .algorithms import *  # noqa: F403
from .algorithms import __all__ as _algorithms
from .comparison import Comparator
from .experiment import Experiment, Report
from .population import Individual, Population
from .problem import Problem, Solution
from .random import StudioRng, derive_seed
from .result import History, Result
from .tuning import tune
from .variables import (
    Binary,
    BinaryPartition,
    BinarySelection,
    FixedSum,
    Integer,
    Partition,
    Permutation,
    Real,
    Selection,
    Variable,
)

__version__ = "0.4.0"

__all__ = [
    "Algorithm", "Binary", "BinaryPartition", "BinarySelection", "Comparator", "Experiment", "FixedSum",
    "History", "Individual", "Integer", "Partition", "Permutation", "Population", "Problem", "Real",
    "Report", "Result", "Selection", "Solution", "StudioRng", "Variable", "benchmarks", "callbacks",
    "derive_seed", "experiment", "operators", "plot", "problems", "random", "tune", "tuning",
    *_algorithms,
]
