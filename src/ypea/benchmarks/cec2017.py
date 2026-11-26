"""CEC 2017 single objective bound constrained benchmark suite.

A port of the official C code ``cec17_test_func.cpp`` (Awad, Ali, Liang, Qu and
Suganthan, 2016), with the official shift vectors, rotation matrices and shuffle data.
The search range is [-100, 100] in every dimension, and function ``f`` has the optimal
value ``100 * f``.

Functions and dimensions follow the C code. F2 was removed from the competition because
it is numerically unstable. It is available on request but left out of ``cec2017(dim)``.

=========  ========================================
dimension  functions
=========  ========================================
2          F1 to F10 and F23 to F28
10, 30,    F1 to F30
50, 100
20         F1 to F10 and F20 to F28
=========  ========================================

The C code differs from the technical report in a few places, and the port follows the
C code:

* F6 (Schaffer F7) uses the shifted point without the rotation.
* F8 (non continuous Rastrigin) is the plain shifted and rotated Rastrigin function.
* F9 (Levy) has its minimum where the rotated point is all ones, at
  ``o + inv(M) @ ones`` instead of at the shift vector ``o``.
* In F14 and F20, the Schaffer F7 part reads the first variables of the whole shuffled
  vector instead of its own segment.

Location of the optimum (``info["x_opt"]``): the shift vector for F1 to F20 except F9,
and the first shift vector of the composition functions F21 to F30.
"""

from __future__ import annotations

import numpy as np

from . import _cec_common as cec

SUITE = "cec2017"
DIMS = (2, 10, 20, 30, 50, 100)
SOURCE = ("CEC 2017 official C code (cec17_test_func.cpp) and input data, "
          "https://github.com/P-N-Suganthan/CEC2017-BoundContrained")
EXCLUDED = (2,)


def _b(name):
    return ("basic", name)


def _h(gp, *names):
    return ("hybrid", gp, names)


_HF05 = _h((0.2, 0.2, 0.3, 0.3), "bent_cigar", "hgbat", "rastrigin", "rosenbrock")
_HF06 = _h((0.2, 0.2, 0.3, 0.3), "escaffer6", "hgbat", "rosenbrock", "schwefel")
_HF07 = _h((0.1, 0.2, 0.2, 0.2, 0.3), "katsuura", "ackley", "grie_rosen", "schwefel", "rastrigin")
_HF08 = _h((0.2,) * 5, "ellips", "ackley", "rastrigin", "hgbat", "discus")
_HF09 = _h((0.2,) * 5, "bent_cigar", "rastrigin", "grie_rosen", "weierstrass", "escaffer6")

# composition components: (spec, mult, div, r_flag), scaled as mult * fit / div
_ELL = (_b("ellips"), 10000, 1e10, 1)
_ACK = (_b("ackley"), 1000, 100, 1)
_GRI = (_b("griewank"), 1000, 100, 1)
_HAP = (_b("happycat"), 1000, 1e3, 1)
_DIS = (_b("discus"), 10000, 1e10, 1)
_ESC = (_b("escaffer6"), 10000, 2e7, 1)


def _p(name):
    return (_b(name), None, None, 1)


def _c(delta, *parts):
    return ("composition", delta, tuple(100.0 * i for i in range(len(delta))), parts)


TABLE = {
    1: _b("bent_cigar"),
    2: _b("sum_diff_pow"),
    3: _b("zakharov"),
    4: _b("rosenbrock"),
    5: _b("rastrigin"),
    6: _b("schaffer_f7"),
    7: _b("bi_rastrigin"),
    8: _b("step_rastrigin"),
    9: _b("levy17"),
    10: _b("schwefel"),
    11: _h((0.2, 0.4, 0.4), "zakharov", "rosenbrock", "rastrigin"),
    12: _h((0.3, 0.3, 0.4), "ellips", "schwefel", "bent_cigar"),
    13: _h((0.3, 0.3, 0.4), "bent_cigar", "rosenbrock", "bi_rastrigin"),
    14: _h((0.2, 0.2, 0.2, 0.4), "ellips", "ackley", "schaffer_f7", "rastrigin"),
    15: _HF05,
    16: _HF06,
    17: _HF07,
    18: _HF08,
    19: _HF09,
    20: _h((0.1, 0.1, 0.2, 0.2, 0.2, 0.2), "hgbat", "katsuura", "ackley", "rastrigin", "schwefel",
           "schaffer_f7"),
    21: _c((10, 20, 30), _p("rosenbrock"), _ELL, _p("rastrigin")),
    22: _c((10, 20, 30), _p("rastrigin"), _GRI, _p("schwefel")),
    23: _c((10, 20, 30, 40), _p("rosenbrock"), _ACK, _p("schwefel"), _p("rastrigin")),
    24: _c((10, 20, 30, 40), _ACK, _ELL, _GRI, _p("rastrigin")),
    25: _c((10, 20, 30, 40, 50), (_b("rastrigin"), 10000, 1e3, 1), _HAP, _ACK, _DIS, _p("rosenbrock")),
    26: _c((10, 20, 20, 30, 40), _ESC, _p("schwefel"), _GRI, _p("rosenbrock"),
           (_b("rastrigin"), 10000, 1e3, 1)),
    27: _c((10, 20, 30, 40, 50, 60), (_b("hgbat"), 10000, 1000, 1), (_b("rastrigin"), 10000, 1e3, 1),
           (_b("schwefel"), 10000, 4e3, 1), (_b("bent_cigar"), 10000, 1e30, 1), _ELL, _ESC),
    28: _c((10, 20, 30, 40, 50, 60), _ACK, _GRI, _DIS, _p("rosenbrock"), _HAP, _ESC),
    29: _c((10, 30, 50), (_HF05, None, None, 1), (_HF06, None, None, 1), (_HF07, None, None, 1)),
    30: _c((10, 30, 50), (_HF05, None, None, 1), (_HF08, None, None, 1), (_HF09, None, None, 1)),
}


def _undefined(f, dim):
    # the C code rejects these at D = 2
    return dim == 2 and (17 <= f <= 22 or f >= 29)


def cec2017_function(f, dim):
    """The array function of CEC 2017 function ``f`` at dimension ``dim``.

    It takes x with shape (..., dim) and returns values with shape (...).
    """
    cec.check(SUITE, TABLE, f, dim, DIMS, _undefined)
    return cec.build(SUITE, f, dim, TABLE[f], 100.0 * f)


def _x_opt(fn):
    if fn.number == 9:
        return fn.shift + np.linalg.solve(fn.M, np.ones(fn.dim))
    return fn.shift[0] if fn.shift.ndim == 2 else fn.shift


def cec2017_problem(f, dim):
    """CEC 2017 function ``f`` at dimension ``dim`` as a vectorized ``Problem``."""
    fn = cec2017_function(f, dim)
    extra = {"excluded": True} if f in EXCLUDED else {}
    return cec.make_problem(fn, _x_opt(fn).copy(), SOURCE, **extra)


def cec2017(dim, functions=None, include_f2=False):
    """CEC 2017 problems at dimension ``dim``.

    Parameters
    ----------
    dim : int
        One of 2, 10, 20, 30, 50 and 100.
    functions : list of int, optional
        Function numbers. By default, every function the C code defines at ``dim``
        except F2. Named functions must exist at ``dim``.
    include_f2 : bool
        Add F2 to the default list. F2 was removed from the competition because it is
        numerically unstable.

    Returns
    -------
    list of Problem
        Each with ``optimum = 100 * f`` and ``info`` with ``suite``, ``function``,
        ``dim``, ``x_opt`` and ``source``.
    """
    if functions is None:
        functions = [f for f in TABLE if cec.supported(SUITE, TABLE, f, dim, DIMS, _undefined)
                     and (include_f2 or f not in EXCLUDED)]
        if not functions:
            raise ValueError(f"CEC 2017 is defined for dimensions {DIMS}, got {dim}")
    return [cec2017_problem(f, dim) for f in functions]
