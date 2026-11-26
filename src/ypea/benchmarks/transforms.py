"""Seeded random shift and rotation for benchmark functions."""

from __future__ import annotations

import numpy as np


def random_shift(lb, ub, rng, fraction=0.8):
    """A random point in the central ``fraction`` of the box [lb, ub].

    With the default 0.8 and bounds [-100, 100] this is [-80, 80], as in the CEC 2017
    technical report. ``rng`` is a NumPy Generator.
    """
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    center, half = (lb + ub) / 2, fraction * (ub - lb) / 2
    return rng.uniform(center - half, center + half)


def random_rotation(d, rng):
    """A random orthogonal d by d matrix, uniformly distributed (Haar measure).

    It is the Q factor of the QR decomposition of a standard Gaussian matrix, with the
    column signs fixed so the diagonal of R is positive (Mezzadri 2007).
    """
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)
