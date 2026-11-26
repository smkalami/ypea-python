"""Classic benchmark functions.

Each function takes one point (shape (d,)) or a batch (shape (..., d)) and returns values
with shape (...), so it can be used directly in vectorized problems. ``REGISTRY`` holds the
standard search bounds, the optimum value and the optimum location of each function.

Sources:

- Jamil and Yang (2013), "A literature survey of benchmark functions for global
  optimization problems", Int. J. Math. Model. Numer. Optim. 4(2).
- Yao, Liu and Lin (1999), "Evolutionary programming made faster", IEEE TEVC 3(2).
- Awad et al. (2016), the CEC 2017 technical report "Problem definitions and evaluation
  criteria for the CEC 2017 special session on single objective real parameter numerical
  optimization".
- Suganthan et al. (2005), the CEC 2005 technical report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np


def _x(x):
    return np.asarray(x, dtype=float)


def _index(x):
    """1 based coordinate index, broadcastable against x."""
    return np.arange(1, x.shape[-1] + 1, dtype=float)


def sphere(x):
    """Sphere: sum x_i^2. Yao et al. (1999) f1."""
    x = _x(x)
    return np.sum(x**2, axis=-1)


def ellipsoid(x):
    """High conditioned elliptic: sum 10^(6 (i-1)/(d-1)) x_i^2. CEC 2017 technical report."""
    x = _x(x)
    d = x.shape[-1]
    w = 10.0 ** (6 * np.arange(d) / (d - 1)) if d > 1 else np.ones(1)
    return np.sum(w * x**2, axis=-1)


def sum_squares(x):
    """Sum of squares: sum i x_i^2. Jamil and Yang (2013)."""
    x = _x(x)
    return np.sum(_index(x) * x**2, axis=-1)


def schwefel_1_2(x):
    """Schwefel 1.2 (double sum): sum_i (sum_{j<=i} x_j)^2. Jamil and Yang (2013)."""
    x = _x(x)
    return np.sum(np.cumsum(x, axis=-1) ** 2, axis=-1)


def schwefel_2_21(x):
    """Schwefel 2.21: max |x_i|. Jamil and Yang (2013)."""
    return np.max(np.abs(_x(x)), axis=-1)


def schwefel_2_22(x):
    """Schwefel 2.22: sum |x_i| + prod |x_i|. Jamil and Yang (2013)."""
    a = np.abs(_x(x))
    return np.sum(a, axis=-1) + np.prod(a, axis=-1)


SCHWEFEL_X = 420.9687463599821
_SCHWEFEL_C = SCHWEFEL_X * np.sin(np.sqrt(SCHWEFEL_X))  # 418.9829...


def schwefel_2_26(x):
    """Schwefel 2.26: sum (c - x_i sin(sqrt|x_i|)) with c = 418.9829..., so the minimum is 0.

    Yao et al. (1999) f8, shifted up by c per coordinate. Inside [-500, 500] it is the
    standard function. Outside, it uses the boundary treatment of the modified Schwefel
    function in the CEC 2017 technical report, so the minimum stays 0 when the function is
    shifted or rotated.
    """
    x = _x(x)
    d = x.shape[-1]
    a = np.abs(x)
    m = 500 - np.mod(a, 500)  # reflection of the part beyond the box
    out = a > 500
    y = np.where(out, np.sign(x) * m, x)
    g = y * np.sin(np.sqrt(np.abs(y))) - np.where(out, (a - 500) ** 2 / (10000 * d), 0.0)
    return np.sum(_SCHWEFEL_C - g, axis=-1)


def rosenbrock(x):
    """Rosenbrock: sum 100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2. Jamil and Yang (2013)."""
    x = _x(x)
    return np.sum(100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1) ** 2, axis=-1)


def rastrigin(x):
    """Rastrigin: sum x_i^2 + 10 (1 - cos(2 pi x_i)). Yao et al. (1999) f9."""
    x = _x(x)
    return np.sum(x**2 + 10 * (1 - np.cos(2 * np.pi * x)), axis=-1)


def ackley(x):
    """Ackley. Yao et al. (1999) f10."""
    x = _x(x)
    return (20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=-1)))
            - np.exp(np.mean(np.cos(2 * np.pi * x), axis=-1)))


def griewank(x):
    """Griewank: sum x_i^2 / 4000 - prod cos(x_i / sqrt(i)) + 1. Jamil and Yang (2013)."""
    x = _x(x)
    return np.sum(x**2, axis=-1) / 4000 - np.prod(np.cos(x / np.sqrt(_index(x))), axis=-1) + 1


def levy(x):
    """Levy, with w_i = 1 + (x_i - 1) / 4. CEC 2017 technical report."""
    w = 1 + (_x(x) - 1) / 4
    wi, wd = w[..., :-1], w[..., -1]
    return (np.sin(np.pi * w[..., 0]) ** 2
            + np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2), axis=-1)
            + (wd - 1) ** 2 * (1 + np.sin(2 * np.pi * wd) ** 2))


def zakharov(x):
    """Zakharov: sum x_i^2 + s^2 + s^4 with s = sum 0.5 i x_i. Jamil and Yang (2013)."""
    x = _x(x)
    s = np.sum(0.5 * _index(x) * x, axis=-1)
    return np.sum(x**2, axis=-1) + s**2 + s**4


def step(x):
    """Step: sum floor(x_i + 0.5)^2. Yao et al. (1999) f6, "Step 2" in Jamil and Yang (2013).

    The minimum 0 is reached on the whole cube [-0.5, 0.5)^d.
    """
    return np.sum(np.floor(_x(x) + 0.5) ** 2, axis=-1)


def quartic(x):
    """Quartic without noise: sum i x_i^4. Yao et al. (1999) f7."""
    x = _x(x)
    return np.sum(_index(x) * x**4, axis=-1)


def quartic_noisy(x, rng):
    """Quartic with noise: sum i x_i^4 + U[0, 1). Yao et al. (1999) f7.

    ``rng`` is a NumPy Generator. One uniform number is drawn per point.
    """
    f = quartic(x)
    return f + rng.random(np.shape(f))


def alpine1(x):
    """Alpine 1: sum |x_i sin(x_i) + 0.1 x_i|. Jamil and Yang (2013)."""
    x = _x(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=-1)


STYBLINSKI_TANG_X = -2.903534027771177


def styblinski_tang(x):
    """Styblinski Tang: 0.5 sum (x_i^4 - 16 x_i^2 + 5 x_i). Jamil and Yang (2013).

    The minimum is about -39.166166 d at x_i = -2.903534.
    """
    x = _x(x)
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x, axis=-1)


def dixon_price(x):
    """Dixon Price: (x_1 - 1)^2 + sum_{i>=2} i (2 x_i^2 - x_{i-1})^2. Jamil and Yang (2013).

    The minimum 0 is at x_i = 2^(-(2^i - 2) / 2^i) (and at the sign variants for i >= 2).
    """
    x = _x(x)
    i = _index(x)[1:]
    return (x[..., 0] - 1) ** 2 + np.sum(i * (2 * x[..., 1:] ** 2 - x[..., :-1]) ** 2, axis=-1)


def michalewicz(x, m=10):
    """Michalewicz: -sum sin(x_i) sin(i x_i^2 / pi)^(2m), with m = 10. Jamil and Yang (2013).

    The literature gives the minimum only for some dimensions (about -1.8013 for d = 2,
    -4.687658 for d = 5 and -9.66015 for d = 10). The function is a sum of one variable
    terms, so ``michalewicz_optimum(d)`` computes the minimizer for any d numerically.
    """
    x = _x(x)
    return -np.sum(np.sin(x) * np.sin(_index(x) * x**2 / np.pi) ** (2 * m), axis=-1)


@cache
def _michalewicz_coordinate(i):
    """Minimizer in [0, pi] of the i-th term of the Michalewicz function (1 based i)."""
    def term(t):
        return -np.sin(t) * np.sin(i * t**2 / np.pi) ** 20

    t = np.linspace(0, np.pi, 200_001)
    k = int(np.argmin(term(t)))
    a, b = t[max(k - 1, 0)], t[min(k + 1, len(t) - 1)]
    r = (np.sqrt(5) - 1) / 2
    for _ in range(80):  # golden section on the bracket around the best grid point
        c, d = b - r * (b - a), a + r * (b - a)
        if term(c) < term(d):
            b = d
        else:
            a = c
    return float((a + b) / 2)


def michalewicz_optimum(d):
    """Minimizer of the Michalewicz function (m = 10) in [0, pi]^d."""
    return np.array([_michalewicz_coordinate(i) for i in range(1, d + 1)])


def salomon(x):
    """Salomon: 1 - cos(2 pi r) + 0.1 r with r = ||x||. Jamil and Yang (2013)."""
    r = np.sqrt(np.sum(_x(x) ** 2, axis=-1))
    return 1 - np.cos(2 * np.pi * r) + 0.1 * r


_WA = 0.5 ** np.arange(21)
_WB = 3.0 ** np.arange(21)


def _weierstrass_sum(x):
    return np.sum(_WA * np.cos(2 * np.pi * _WB * (x[..., None] + 0.5)), axis=-1)


def weierstrass(x):
    """Weierstrass with a = 0.5, b = 3 and k_max = 20. CEC 2017 technical report."""
    x = _x(x)
    # subtract the constant per coordinate, so the terms cancel exactly at x = 0
    return np.sum(_weierstrass_sum(x) - _weierstrass_sum(np.zeros(1)), axis=-1)


def happy_cat(x):
    """HappyCat: |S - d|^(1/4) + (0.5 S + sum x_i) / d + 0.5 with S = sum x_i^2.

    CEC 2017 technical report (Beyer and Finck 2012). The minimum 0 is at x_i = -1.
    """
    x = _x(x)
    d = x.shape[-1]
    s2, s1 = np.sum(x**2, axis=-1), np.sum(x, axis=-1)
    return np.abs(s2 - d) ** 0.25 + (0.5 * s2 + s1) / d + 0.5


def hgbat(x):
    """HGBat: |S^2 - T^2|^(1/2) + (0.5 S + T) / d + 0.5 with S = sum x_i^2, T = sum x_i.

    CEC 2017 technical report. The minimum 0 is at x_i = -1.
    """
    x = _x(x)
    d = x.shape[-1]
    s2, s1 = np.sum(x**2, axis=-1), np.sum(x, axis=-1)
    return np.abs(s2**2 - s1**2) ** 0.5 + (0.5 * s2 + s1) / d + 0.5


def expanded_schaffer_f6(x):
    """Expanded Schaffer F6: sum g(x_i, x_{i+1}) over the ring of coordinates, with
    g(x, y) = 0.5 + (sin^2(sqrt(x^2 + y^2)) - 0.5) / (1 + 0.001 (x^2 + y^2))^2.

    CEC 2017 technical report.
    """
    x = _x(x)
    r2 = x**2 + np.roll(x, -1, axis=-1) ** 2
    return np.sum(0.5 + (np.sin(np.sqrt(r2)) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2, axis=-1)


def bent_cigar(x):
    """Bent cigar: x_1^2 + 10^6 sum_{i>=2} x_i^2. CEC 2017 technical report."""
    x = _x(x)
    return x[..., 0] ** 2 + 1e6 * np.sum(x[..., 1:] ** 2, axis=-1)


def discus(x):
    """Discus: 10^6 x_1^2 + sum_{i>=2} x_i^2. CEC 2017 technical report."""
    x = _x(x)
    return 1e6 * x[..., 0] ** 2 + np.sum(x[..., 1:] ** 2, axis=-1)


@dataclass(frozen=True)
class Benchmark:
    """Registry entry: the array function and its standard setting.

    ``x_opt`` is a number (the same for every coordinate) or a function of d. ``f_opt`` is
    a number, a function of d, or None to use f(x_opt). ``transformable`` is False when the
    optimum holds only inside the standard bounds, so shift and rotation are not allowed.
    """

    fn: Callable
    lb: float
    ub: float
    x_opt: object
    f_opt: object
    source: str
    transformable: bool = True

    def optimum_x(self, d):
        x = self.x_opt(d) if callable(self.x_opt) else np.full(d, float(self.x_opt))
        return np.asarray(x, dtype=float)

    def optimum_f(self, d):
        if self.f_opt is None:
            return float(self.fn(self.optimum_x(d)))
        return float(self.f_opt(d) if callable(self.f_opt) else self.f_opt)


def _dixon_price_x(d):
    i = np.arange(1, d + 1, dtype=float)
    return 2.0 ** (-(2.0**i - 2) / 2.0**i)


def _styblinski_tang_f(d):
    return d * float(styblinski_tang(np.array([STYBLINSKI_TANG_X])))


_JY = "Jamil and Yang (2013)"
_YAO = "Yao, Liu and Lin (1999)"
_CEC = "CEC 2017 technical report (Awad et al. 2016)"

REGISTRY = {
    "sphere": Benchmark(sphere, -100, 100, 0, 0, _YAO),
    "ellipsoid": Benchmark(ellipsoid, -100, 100, 0, 0, _CEC),
    "sum_squares": Benchmark(sum_squares, -10, 10, 0, 0, _JY),
    "schwefel_1_2": Benchmark(schwefel_1_2, -100, 100, 0, 0, _JY),
    "schwefel_2_21": Benchmark(schwefel_2_21, -100, 100, 0, 0, _JY),
    "schwefel_2_22": Benchmark(schwefel_2_22, -10, 10, 0, 0, _JY),
    "schwefel_2_26": Benchmark(schwefel_2_26, -500, 500, SCHWEFEL_X, 0, _YAO + ", " + _CEC),
    "rosenbrock": Benchmark(rosenbrock, -30, 30, 1, 0, _JY),
    "rastrigin": Benchmark(rastrigin, -5.12, 5.12, 0, 0, _YAO),
    "ackley": Benchmark(ackley, -32.768, 32.768, 0, 0, _YAO),
    "griewank": Benchmark(griewank, -600, 600, 0, 0, _JY),
    "levy": Benchmark(levy, -10, 10, 1, 0, _CEC),
    "zakharov": Benchmark(zakharov, -5, 10, 0, 0, _JY),
    "step": Benchmark(step, -100, 100, 0, 0, _YAO + ", " + _JY),
    "quartic": Benchmark(quartic, -1.28, 1.28, 0, 0, _YAO),
    "alpine1": Benchmark(alpine1, -10, 10, 0, 0, _JY),
    "styblinski_tang": Benchmark(styblinski_tang, -5, 5, STYBLINSKI_TANG_X, _styblinski_tang_f, _JY),
    "dixon_price": Benchmark(dixon_price, -10, 10, _dixon_price_x, 0, _JY),
    # outside [0, pi] the terms reach lower values, so shift and rotation would move the optimum
    "michalewicz": Benchmark(michalewicz, 0, np.pi, michalewicz_optimum, None, _JY, transformable=False),
    "salomon": Benchmark(salomon, -100, 100, 0, 0, _JY),
    "weierstrass": Benchmark(weierstrass, -0.5, 0.5, 0, 0, _CEC + ", bounds from CEC 2005"),
    "happy_cat": Benchmark(happy_cat, -100, 100, -1, 0, _CEC),
    "hgbat": Benchmark(hgbat, -100, 100, -1, 0, _CEC),
    "expanded_schaffer_f6": Benchmark(expanded_schaffer_f6, -100, 100, 0, 0, _CEC),
    "bent_cigar": Benchmark(bent_cigar, -100, 100, 0, 0, _CEC),
    "discus": Benchmark(discus, -100, 100, 0, 0, _CEC),
}
