"""Module level objectives, so problems built from them can be pickled."""

import numpy as np

from ypea.benchmarks import sphere


def sphere_obj(sol):
    return sphere(sol.x)


def neg_sphere_obj(sol):
    return -sphere(sol.x)


def constrained_obj(sol):
    # minimize (x - 2)^2 + (y - 1)^2 subject to x + y <= 2: optimum (1.5, 0.5), f = 0.5
    x, y = sol.x[..., 0], sol.x[..., 1]
    return (x - 2) ** 2 + (y - 1) ** 2, x + y - 2


def mixed_obj(sol):
    # touches every variable type, so smoke tests decode all of them
    total = float(np.sum(sol.r**2) + np.sum(sol.i) + np.sum(sol.b))
    total += float(np.sum(sol.p * np.arange(len(sol.p))))
    total += float(np.sum(sol.s) + np.sum(sol.bs * np.arange(len(sol.bs))))
    total += float(sum(len(g) ** 2 for g in sol.q))
    total += float(np.sum(sol.bp[0]) + np.sum(sol.fs**2))
    return total
