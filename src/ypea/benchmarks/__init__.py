"""Benchmark functions and problems.

Array functions (``sphere``, ``rastrigin`` and so on) take x with shape (..., d) and return
values with shape (...), so they work in vectorized problems. Problem factories:

* ``problem(name, dim, shift=False, rotate=False, seed=0)`` and ``classic(dim)`` for the
  classic functions, with optional seeded shift and rotation.
* ``cec2017(dim)`` and ``cec2022(dim)`` for the CEC suites (official data, checked against
  the official C code). ``cec2017_function`` and ``cec2022_function`` give the raw functions.
"""

from .cec2017 import cec2017, cec2017_function, cec2017_problem  # noqa: E402,I001
from .cec2022 import cec2022, cec2022_function, cec2022_problem  # noqa: E402,I001
from .classic import (
    REGISTRY,
    ackley,
    alpine1,
    bent_cigar,
    discus,
    dixon_price,
    ellipsoid,
    expanded_schaffer_f6,
    griewank,
    happy_cat,
    hgbat,
    levy,
    michalewicz,
    quartic,
    quartic_noisy,
    rastrigin,
    rosenbrock,
    salomon,
    schwefel_1_2,
    schwefel_2_21,
    schwefel_2_22,
    schwefel_2_26,
    sphere,
    step,
    styblinski_tang,
    sum_squares,
    weierstrass,
    zakharov,
)
from .problems import classic, names, problem  # after .classic: the suite replaces the submodule name

__all__ = [
    "REGISTRY", "ackley", "alpine1", "bent_cigar", "cec2017", "cec2017_function", "cec2017_problem",
    "cec2022", "cec2022_function", "cec2022_problem", "classic", "discus", "dixon_price", "ellipsoid",
    "expanded_schaffer_f6", "griewank", "happy_cat", "hgbat", "levy", "michalewicz", "names", "problem",
    "quartic", "quartic_noisy", "rastrigin", "rosenbrock", "salomon", "schwefel_1_2", "schwefel_2_21",
    "schwefel_2_22", "schwefel_2_26", "sphere", "step", "styblinski_tang", "sum_squares", "weierstrass",
    "zakharov",
]
