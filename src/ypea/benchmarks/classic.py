"""Classic benchmark functions (minimum value 0).

They work on one point (shape (d,)) or a batch (shape (n, d)), so they can be used
directly in vectorized problems.
"""

from __future__ import annotations

import numpy as np


def sphere(x):
    x = np.asarray(x, dtype=float)
    return np.sum(x**2, axis=-1)


def ackley(x):
    x = np.asarray(x, dtype=float)
    return (20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=-1)))
            - np.exp(np.mean(np.cos(2 * np.pi * x), axis=-1)))


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1) ** 2, axis=-1)


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return np.sum(x**2 + 10 * (1 - np.cos(2 * np.pi * x)), axis=-1)
