"""Shared parts of the CEC 2017 and CEC 2022 suites.

The functions are ported from the official C code (``cec17_test_func.cpp`` and
``cec22_test_func.cpp``) and keep its exact behavior, including the details that differ
from the technical reports. Every function works on a batch ``x`` with shape (n, d).

The C functions share two global buffers, ``y`` and ``z``, and a few functions read the
wrong one. Those cases are reproduced on purpose and marked in the comments below.
"""

from __future__ import annotations

import math
from functools import cache
from importlib import resources

import numpy as np

from ..problem import Problem
from ..variables import Real

INF = 1.0e99  # the C code uses this finite value, not infinity
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029


def _read(suite, keys=None):
    path = resources.files("ypea.benchmarks").joinpath("data", f"{suite}.npz")
    with path.open("rb") as fh, np.load(fh) as data:
        return data.files if keys is None else [data[k] for k in keys]


@cache
def data_files(suite):
    """Names of the arrays in the bundled data of a suite."""
    return frozenset(_read(suite))


def sr(x, o, M, rate, s_flag, r_flag):
    """Shift, shrink and rotate, as ``sr_func`` does. Returns (z, y), where y is the value
    before the rotation (the C buffer ``y`` when it is written)."""
    y = (x - o) * rate if s_flag else x * rate
    # C: z[i] = sum_j y[j] * M[i*nx + j], with M stored row major
    return (y @ M.T if r_flag else y), y


# basic functions of the transformed point z, shape (n, k)

def bent_cigar(z):
    return z[:, 0] ** 2 + np.sum(1e6 * z[:, 1:] ** 2, axis=1)


def discus(z):
    return 1e6 * z[:, 0] ** 2 + np.sum(z[:, 1:] ** 2, axis=1)


def ellips(z):
    k = z.shape[1]
    return np.sum(10.0 ** (6.0 * np.arange(k) / (k - 1)) * z**2, axis=1)


def sum_diff_pow(z):
    # C calls abs() on a double here, whose meaning depends on the compiler, so fabs is used
    return np.sum(np.abs(z) ** np.arange(1, z.shape[1] + 1), axis=1)


def zakharov(z):
    s1 = np.sum(z**2, axis=1)
    s2 = np.sum(0.5 * np.arange(1, z.shape[1] + 1) * z, axis=1)
    return s1 + s2**2 + s2**4


def _levy(z, c):
    w = 1.0 + (z - c) / 4.0
    term1 = np.sin(PI * w[:, 0]) ** 2
    term3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * PI * w[:, -1]) ** 2)
    wi = w[:, :-1]
    return term1 + np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(PI * wi + 1) ** 2), axis=1) + term3


def levy17(z):
    # CEC 2017 uses w = 1 + (z - 1)/4, so the minimum is at z = 1, not at z = 0
    return _levy(z, 1.0)


def levy22(z):
    return _levy(z, 0.0)


def rosenbrock(z):
    z = z + 1.0
    return np.sum(100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1.0) ** 2, axis=1)


def schaffer_f7(y):
    k = y.shape[1]
    s = (y[:, :-1] ** 2 + y[:, 1:] ** 2) ** 0.5
    t = np.sin(50.0 * s**0.2)
    f = np.sum(s**0.5 + s**0.5 * t * t, axis=1)
    return f * f / (k - 1) / (k - 1)


def ackley(z):
    k = z.shape[1]
    s1 = -0.2 * np.sqrt(np.sum(z**2, axis=1) / k)
    s2 = np.sum(np.cos(2.0 * PI * z), axis=1) / k
    return E - 20.0 * np.exp(s1) - np.exp(s2) + 20.0


_WA = 0.5 ** np.arange(21)
_WB = 3.0 ** np.arange(21)


def weierstrass(z):
    k = z.shape[1]
    s = np.sum(_WA * np.cos(2.0 * PI * _WB * (z[..., None] + 0.5)), axis=2)
    s2 = np.sum(_WA * np.cos(2.0 * PI * _WB * 0.5))
    return np.sum(s, axis=1) - k * s2


def griewank(z):
    p = np.prod(np.cos(z / np.sqrt(1.0 + np.arange(z.shape[1]))), axis=1)
    return 1.0 + np.sum(z**2, axis=1) / 4000.0 - p


def rastrigin(z):
    return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0, axis=1)


def schwefel(z):
    k = z.shape[1]
    z = z + 4.209687462275036e002
    m = np.fmod(np.abs(z), 500)
    high = -(500.0 - m) * np.sin((500.0 - m) ** 0.5) + ((z - 500.0) / 100) ** 2 / k
    low = -(-500.0 + m) * np.sin((500.0 - m) ** 0.5) + ((z + 500.0) / 100) ** 2 / k
    mid = -z * np.sin(np.abs(z) ** 0.5)
    f = np.where(z > 500, high, np.where(z < -500, low, mid))
    return np.sum(f, axis=1) + 4.189828872724338e002 * k


_KP = 2.0 ** np.arange(1, 33)


def katsuura(z):
    k = z.shape[1]
    t = _KP * z[..., None]
    temp = np.sum(np.abs(t - np.floor(t + 0.5)) / _KP, axis=2)
    f = np.prod((1.0 + np.arange(1, k + 1) * temp) ** (10.0 / k**1.2), axis=1)
    c = 10.0 / k / k
    return f * c - c


def grie_rosen(z):
    z = z + 1.0
    zn = np.roll(z, -1, axis=1)  # the last pair wraps to the first variable
    t = 100.0 * (z**2 - zn) ** 2 + (z - 1.0) ** 2
    return np.sum(t * t / 4000.0 - np.cos(t) + 1.0, axis=1)


def escaffer6(z):
    t = z**2 + np.roll(z, -1, axis=1) ** 2
    return np.sum(0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1.0 + 0.001 * t) ** 2, axis=1)


def happycat(z):
    k = z.shape[1]
    z = z - 1.0
    r2 = np.sum(z * z, axis=1)
    s = np.sum(z, axis=1)
    return np.abs(r2 - k) ** 0.25 + (0.5 * r2 + s) / k + 0.5


def hgbat(z):
    k = z.shape[1]
    z = z - 1.0
    r2 = np.sum(z * z, axis=1)
    s = np.sum(z, axis=1)
    return np.abs(r2**2.0 - s**2.0) ** 0.5 + (0.5 * r2 + s) / k + 0.5


# name: (function of z, shrink rate applied before the rotation)
BASIC = {
    "bent_cigar": (bent_cigar, 1.0),
    "discus": (discus, 1.0),
    "ellips": (ellips, 1.0),
    "sum_diff_pow": (sum_diff_pow, 1.0),
    "zakharov": (zakharov, 1.0),
    "levy17": (levy17, 1.0),
    "levy22": (levy22, 1.0),
    "rosenbrock": (rosenbrock, 2.048 / 100.0),
    "ackley": (ackley, 1.0),
    "weierstrass": (weierstrass, 0.5 / 100.0),
    "griewank": (griewank, 600.0 / 100.0),
    "rastrigin": (rastrigin, 5.12 / 100.0),
    # The C pre-loop of step_rastrigin_func edits the buffer y, which sr_func then
    # overwrites, so the function is plain Rastrigin.
    "step_rastrigin": (rastrigin, 5.12 / 100.0),
    "schwefel": (schwefel, 1000.0 / 100.0),
    "katsuura": (katsuura, 5.0 / 100.0),
    "grie_rosen": (grie_rosen, 5.0 / 100.0),
    "escaffer6": (escaffer6, 1.0),
    "happycat": (happycat, 5.0 / 100.0),
    "hgbat": (hgbat, 5.0 / 100.0),
}


def bi_rastrigin(x, o, M, s_flag, r_flag):
    """Lunacek bi-Rastrigin. The sign flip uses ``o[:k]`` even inside a hybrid function."""
    k = x.shape[1]
    mu0, d = 2.5, 1.0
    s = 1.0 - 1.0 / (2.0 * (k + 20.0) ** 0.5 - 8.2)
    mu1 = -(((mu0 * mu0 - d) / s) ** 0.5)
    y = (x - o) if s_flag else x
    y = y * (10.0 / 100.0)
    z = np.where(o[:k] < 0.0, -(2 * y), 2 * y)
    t = z + mu0
    tmp1 = np.sum((t - mu0) ** 2, axis=1)
    tmp2 = np.sum((t - mu1) ** 2, axis=1) * s + d * k
    c = np.sum(np.cos(2.0 * PI * (z @ M.T if r_flag else z)), axis=1)
    return np.minimum(tmp1, tmp2) + 10.0 * (k - c)


def basic(name, x, o, M, s_flag, r_flag):
    if name == "bi_rastrigin":
        return bi_rastrigin(x, o, M, s_flag, r_flag)
    if name == "schaffer_f7":
        # The C code reads the buffer y (shifted, not rotated) instead of z, so the
        # rotation has no effect. It is always called here with both flags set.
        return schaffer_f7(sr(x, o, M, 1.0, s_flag, r_flag)[1])
    f, rate = BASIC[name]
    return f(sr(x, o, M, rate, s_flag, r_flag)[0])


def hybrid_sizes(gp, nx):
    sizes = [int(math.ceil(g * nx)) for g in gp[:-1]]  # float product, as in C
    return [*sizes, nx - sum(sizes)]


def hybrid(gp, names, x, o, M, S, s_flag=1, r_flag=1):
    nx = x.shape[1]
    y = sr(x, o, M, 1.0, s_flag, r_flag)[0][:, S - 1]  # the shuffle data is 1 based
    f, g = 0.0, 0
    for name, k in zip(names, hybrid_sizes(gp, nx), strict=True):
        if name == "schaffer_f7":
            # C reads y[0..k) of the whole shuffled vector, not its own segment
            fit = schaffer_f7(y[:, :k])
        else:
            fit = basic(name, y[:, g:g + k], o[:k], M, 0, 0)
        f = f + fit
        g += k
    return f


def cf_cal(x, shifts, delta, bias, fit):
    """Composition weights, as ``cf_cal`` computes them. ``shifts`` has one row per component."""
    nx = x.shape[1]
    fit = fit + np.asarray(bias, dtype=float)
    w = np.sum((x[:, None, :] - shifts[None]) ** 2.0, axis=2)
    safe = np.where(w != 0, w, 1.0)
    delta = np.asarray(delta, dtype=float)
    w = np.where(w != 0, (1.0 / safe) ** 0.5 * np.exp(-safe / 2.0 / nx / delta**2.0), INF)
    w_sum = np.sum(w, axis=1, keepdims=True)
    zero = np.max(w, axis=1) == 0
    w[zero] = 1.0
    w_sum[zero] = len(delta)
    return np.sum(w / w_sum * fit, axis=1)


class CECFunction:
    """One CEC function at one dimension, an array function of x with shape (..., dim).

    ``spec`` is a tuple from the suite table:
    ``("basic", name)``, ``("hybrid", gp, names)`` or
    ``("composition", delta, bias, components)``, where each component is
    ``(spec, mult, div, r_flag)`` and its value is scaled as ``mult * fit / div``.
    """

    def __init__(self, suite, number, dim, spec, bias, shift, M, S=None):
        self.suite, self.number, self.dim = suite, number, dim
        self.spec, self.bias = spec, float(bias)
        self.shift, self.M, self.S = shift, M, S
        self.__name__ = f"{suite}_f{number}_d{dim}"

    def __repr__(self):
        return f"CECFunction({self.__name__})"

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1:] != (self.dim,):
            raise ValueError(f"{self.__name__} expects points with {self.dim} values, got shape {x.shape}")
        X = x.reshape(-1, self.dim)
        with np.errstate(over="ignore"):  # large powers overflow to inf, as in C
            f = _evaluate(self.spec, X, self.shift, self.M, self.S, 1) + self.bias
        return f.reshape(x.shape[:-1])


def _evaluate(spec, x, o, M, S, r_flag):
    kind = spec[0]
    if kind == "basic":
        return basic(spec[1], x, o, M, 1, r_flag)
    if kind == "hybrid":
        return hybrid(spec[1], spec[2], x, o, M, S, 1, r_flag)
    _, delta, bias, components = spec
    fits = []
    for i, (sub, mult, div, rf) in enumerate(components):
        fit = _evaluate(sub, x, o[i], M[i], None if S is None else S[i], rf)
        fits.append(fit if mult is None else mult * fit / div)
    return cf_cal(x, o[:len(components)], delta, bias, np.stack(fits, axis=1))


def components(spec):
    return len(spec[3]) if spec[0] == "composition" else 1


def build(suite, number, dim, spec, bias):
    """Slice the official data the way the C code reads it and return a CECFunction."""
    c = components(spec)
    keys = required_keys(number, dim, spec)
    shift, M, *rest = _read(suite, keys)  # read only when a function is created
    shift, M = shift[:, :dim], M.ravel()
    if spec[0] == "composition":
        shift, M = shift[:c], M[:c * dim * dim].reshape(c, dim, dim)
    else:
        shift, M = shift[0], M[:dim * dim].reshape(dim, dim)
    S = None
    if rest:
        S = rest[0].ravel()
        S = S[:c * dim].reshape(c, dim) if spec[0] == "composition" else S[:dim]
    return CECFunction(suite, number, dim, spec, bias, shift.copy(), M.copy(), S)


def _uses_shuffle(spec):
    if spec[0] == "hybrid":
        return True
    return spec[0] == "composition" and any(sub[0] == "hybrid" for sub, *_ in spec[3])


def required_keys(number, dim, spec):
    keys = [f"shift_data_{number}", f"M_{number}_D{dim}"]
    if _uses_shuffle(spec):
        keys.append(f"shuffle_data_{number}_D{dim}")
    return keys


class VectorObjective:
    """Objective of a vectorized problem with one variable ``x``."""

    def __init__(self, function):
        self.function = function
        self.__name__ = function.__name__

    def __call__(self, sol):
        return self.function(sol.x)


def make_problem(function, x_opt, source, **info):
    return Problem(
        Real("x", shape=function.dim, lb=-100, ub=100),
        VectorObjective(function),
        vectorized=True,
        name=function.__name__,
        optimum=function.bias,
        info={"suite": function.suite, "function": function.number, "dim": function.dim,
              "x_opt": x_opt, "source": source, **info},
    )


def supported(suite, table, number, dim, dims, undefined):
    """True if the C code defines function ``number`` at ``dim`` and the data exists."""
    if number not in table or dim not in dims or undefined(number, dim):
        return False
    files = data_files(suite)
    return all(k in files for k in required_keys(number, dim, table[number]))


def check(suite, table, number, dim, dims, undefined):
    if not supported(suite, table, number, dim, dims, undefined):
        raise ValueError(f"{suite} has no function {number} at dimension {dim}")
