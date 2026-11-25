"""Plots for single runs. Needs matplotlib (``pip install ypea[plot]``).

The experiment plots (several runs and algorithms) will live in ``ypea.experiment``.
"""

from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install ypea[plot]") from None
    return plt


def convergence(*results, x="nfe", log=None, ax=None, labels=None, error=False):
    """Best objective value against evaluations (``x="nfe"``) or iterations (``x="iter"``).

    Several results can be drawn on the same axes. With ``error=True`` the distance to the
    known optimum is drawn. ``log`` defaults to True when all values are positive.
    Returns the matplotlib Axes.
    """
    plt = _plt()
    ax = ax or plt.figure(figsize=(6, 4)).gca()
    all_y = []
    for k, r in enumerate(results):
        h = r.history
        if x == "nfe":
            xs, ys = np.r_[h.trace_nfe, r.nfe], np.r_[h.trace_obj_value, h.trace_obj_value[-1]]
            draw = ax.step
        elif x in ("iter", "iteration"):
            xs, ys, draw = h.iteration, h.best_obj_value, ax.plot
        else:
            raise ValueError("x must be 'nfe' or 'iter'")
        if error:
            if r.optimum is None:
                raise ValueError("error=True needs a problem with a known optimum")
            ys = np.abs(ys - r.optimum)
        label = labels[k] if labels else r.algorithm
        if draw is ax.step:
            draw(xs, ys, where="post", label=label)
        else:
            draw(xs, ys, label=label)
        all_y.append(ys)
    y = np.concatenate(all_y)
    if log is None:
        log = bool(np.all(y[np.isfinite(y)] > 0))
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Function evaluations" if x == "nfe" else "Iteration")
    ax.set_ylabel("Error" if error else "Best objective value")
    if len(results) > 1 or labels:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
