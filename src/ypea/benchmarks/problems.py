"""Ready made ``Problem`` objects for the classic benchmark functions."""

from __future__ import annotations

import numpy as np

from ..problem import Problem
from ..variables import Real
from .classic import REGISTRY
from .transforms import random_rotation, random_shift


class Transformed:
    """Objective ``f(z)`` with ``z = z_opt + R (x - x_opt)``.

    ``x_opt`` is the optimum in the search space and ``z_opt`` the optimum of the base
    function, so ``f(x_opt)`` is exactly the base optimum value. ``R`` may be None (no
    rotation). With ``x_opt`` None the base function gets ``x`` unchanged.
    """

    def __init__(self, fn, x_opt=None, z_opt=None, R=None):
        self.fn, self.x_opt, self.z_opt, self.R = fn, x_opt, z_opt, R

    def __call__(self, sol):
        x = np.asarray(sol.x, dtype=float)
        if self.x_opt is None:
            return self.fn(x)
        y = x - self.x_opt
        if self.R is not None:
            y = y @ self.R.T
        return self.fn(self.z_opt + y)


def names():
    """Names of the classic benchmark functions."""
    return list(REGISTRY)


def problem(name, dim, shift=False, rotate=False, seed=0, bounds=None):
    """A classic benchmark as a vectorized ``Problem`` with one variable ``x``.

    Parameters
    ----------
    name : str
        One of ``names()``.
    dim : int
        Number of dimensions.
    shift : bool
        Move the optimum to a random point in the central 80 percent of the bounds.
    rotate : bool
        Rotate the function around its optimum with a random orthogonal matrix.
    seed : int
        Seed of the shift and the rotation. The shift does not depend on ``rotate`` and the
        rotation does not depend on ``shift``.
    bounds : (lb, ub), optional
        Search bounds instead of the standard ones.

    ``info`` holds ``x_opt`` (the optimum location), ``bounds``, ``source``, ``shift``
    (the vector ``o`` such that ``z = R (x - o)`` is the base function argument, or None),
    ``rotate`` (the matrix or None) and ``seed``.
    """
    if name not in REGISTRY:
        raise ValueError(f"unknown benchmark {name!r}, choose from {names()}")
    b = REGISTRY[name]
    dim = int(dim)
    if dim < 1 or (dim < 2 and name in ("rosenbrock", "dixon_price", "levy")):
        raise ValueError(f"{name}: dimension {dim} is too small")
    if (shift or rotate) and not b.transformable:
        raise ValueError(f"{name}: the optimum is known only inside the standard bounds, "
                         "so it cannot be shifted or rotated")
    lb, ub = (b.lb, b.ub) if bounds is None else bounds
    lb = np.broadcast_to(np.asarray(lb, dtype=float), dim).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), dim).copy()
    z_opt = b.optimum_x(dim)
    x_opt, R, o = z_opt, None, None
    if shift:
        x_opt = random_shift(lb, ub, np.random.default_rng([seed, 1]))
    if rotate:
        R = random_rotation(dim, np.random.default_rng([seed, 2]))
    if shift or rotate:
        objective = Transformed(b.fn, x_opt, z_opt, R)
        o = x_opt - (z_opt if R is None else R.T @ z_opt)
    else:
        objective = Transformed(b.fn)
    tag = f"{name}_d{dim}" + "_shifted" * bool(shift) + "_rotated" * bool(rotate)
    if shift or rotate:
        tag += f"_s{seed}"
    info = {"x_opt": x_opt.copy(), "bounds": (lb, ub), "source": b.source,
            "shift": o, "rotate": R, "seed": seed if (shift or rotate) else None}
    return Problem(Real("x", shape=dim, lb=lb, ub=ub), objective, vectorized=True,
                   name=tag, optimum=b.optimum_f(dim), info=info)


def classic(dim, names=None, shift=False, rotate=False, seed=0):
    """A list of classic benchmark problems (all of them by default).

    With ``shift`` or ``rotate`` and no ``names``, functions that cannot be transformed
    (Michalewicz) are left out.
    """
    if names is None:
        names = [n for n, b in REGISTRY.items() if b.transformable or not (shift or rotate)]
    return [problem(n, dim, shift=shift, rotate=rotate, seed=seed) for n in names]
