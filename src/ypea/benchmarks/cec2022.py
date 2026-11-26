"""CEC 2022 single objective bound constrained benchmark suite.

A port of the official C code ``cec22_test_func.cpp`` (Kumar, Price, Mohamed, Hadi and
Suganthan, 2021), with the official shift vectors, rotation matrices and shuffle data.
The search range is [-100, 100] in every dimension. F1 to F12 are defined at dimensions
10 and 20, and all but F6, F7 and F8 also at dimension 2.

Optimal values: 300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600 and 2700.

The C code differs from the technical report in a few places, and the port follows the
C code:

* F3 (Schaffer F7, named expanded Schaffer f6 in the report) uses the shifted point
  without the rotation.
* F4 (non continuous Rastrigin) is the plain shifted and rotated Rastrigin function.
* In F7, the Schaffer F7 part reads the first variables of the whole shuffled vector
  instead of its own segment.
* F10 has three components, not four.

Location of the optimum (``info["x_opt"]``): the shift vector for F1 to F8, and the first
shift vector of the composition functions F9 to F12.
"""

from __future__ import annotations

from . import _cec_common as cec

SUITE = "cec2022"
DIMS = (2, 10, 20)
SOURCE = ("CEC 2022 official C code (cec22_test_func.cpp) and input data, "
          "https://github.com/P-N-Suganthan/2022-SO-BO")
BIAS = {1: 300, 2: 400, 3: 600, 4: 800, 5: 900, 6: 1800, 7: 2000, 8: 2200,
        9: 2300, 10: 2400, 11: 2600, 12: 2700}


def _b(name):
    return ("basic", name)


def _h(gp, *names):
    return ("hybrid", gp, names)


TABLE = {
    1: _b("zakharov"),
    2: _b("rosenbrock"),
    3: _b("schaffer_f7"),
    4: _b("step_rastrigin"),
    5: _b("levy22"),
    6: _h((0.4, 0.4, 0.2), "bent_cigar", "hgbat", "rastrigin"),
    7: _h((0.1, 0.2, 0.2, 0.2, 0.1, 0.2), "hgbat", "katsuura", "ackley", "rastrigin", "schwefel",
          "schaffer_f7"),
    8: _h((0.3, 0.2, 0.2, 0.1, 0.2), "katsuura", "happycat", "grie_rosen", "schwefel", "ackley"),
    # composition components: (spec, mult, div, r_flag), scaled as mult * fit / div
    9: ("composition", (10, 20, 30, 40, 50), (0, 200, 300, 100, 400), (
        (_b("rosenbrock"), 10000, 1e4, 1),
        (_b("ellips"), 10000, 1e10, 1),
        (_b("bent_cigar"), 10000, 1e30, 1),
        (_b("discus"), 10000, 1e10, 1),
        (_b("ellips"), 10000, 1e10, 0))),
    10: ("composition", (20, 10, 10), (0, 200, 100), (
        (_b("schwefel"), None, None, 0),
        (_b("rastrigin"), None, None, 1),
        (_b("hgbat"), None, None, 1))),
    11: ("composition", (20, 20, 30, 30, 20), (0, 200, 300, 400, 200), (
        (_b("escaffer6"), 10000, 2e7, 1),
        (_b("schwefel"), None, None, 1),
        (_b("griewank"), 1000, 100, 1),
        (_b("rosenbrock"), None, None, 1),
        (_b("rastrigin"), 10000, 1e3, 1))),
    12: ("composition", (10, 20, 30, 40, 50, 60), (0, 300, 500, 100, 400, 200), (
        (_b("hgbat"), 10000, 1000, 1),
        (_b("rastrigin"), 10000, 1e3, 1),
        (_b("schwefel"), 10000, 4e3, 1),
        (_b("bent_cigar"), 10000, 1e30, 1),
        (_b("ellips"), 10000, 1e10, 1),
        (_b("escaffer6"), 10000, 2e7, 1))),
}


def _undefined(f, dim):
    # the C code rejects these at D = 2
    return dim == 2 and 6 <= f <= 8


def cec2022_function(f, dim):
    """The array function of CEC 2022 function ``f`` at dimension ``dim``.

    It takes x with shape (..., dim) and returns values with shape (...).
    """
    cec.check(SUITE, TABLE, f, dim, DIMS, _undefined)
    return cec.build(SUITE, f, dim, TABLE[f], BIAS[f])


def cec2022_problem(f, dim):
    """CEC 2022 function ``f`` at dimension ``dim`` as a vectorized ``Problem``."""
    fn = cec2022_function(f, dim)
    x_opt = fn.shift[0] if fn.shift.ndim == 2 else fn.shift
    return cec.make_problem(fn, x_opt.copy(), SOURCE)


def cec2022(dim, functions=None):
    """CEC 2022 problems at dimension ``dim`` (2, 10 or 20).

    ``functions`` is a list of function numbers. By default, every function the C code
    defines at ``dim``. Each problem has ``optimum`` set to the function's optimal value
    and ``info`` with ``suite``, ``function``, ``dim``, ``x_opt`` and ``source``.
    """
    if functions is None:
        functions = [f for f in TABLE if cec.supported(SUITE, TABLE, f, dim, DIMS, _undefined)]
        if not functions:
            raise ValueError(f"CEC 2022 is defined for dimensions {DIMS}, got {dim}")
    return [cec2022_problem(f, dim) for f in functions]
