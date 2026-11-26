"""Plots. Needs matplotlib (``pip install ypea[plot]``).

``convergence`` draws single runs and ``front`` the objective vectors of multi-objective
results. ``band``, ``boxes`` and ``rank_bars`` draw the results
of an experiment (``ypea.experiment.Report`` calls them). Experiment figures are sized
for one column of a paper (3.5 by 2.6 inches, 8 point text). Save them with
``ax.figure.savefig("name.pdf")`` (or .png, .svg).
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
        if r.best is None:
            raise ValueError("a multi-objective result has no best value to draw: plot its front instead")
        h = r.history
        if x == "nfe":
            xs, ys = np.r_[h.trace_nfe, r.nfe], np.r_[h.trace_obj_value, h.trace_obj_value[-1]]
        elif x in ("iter", "iteration"):
            xs, ys = h.iteration, h.best_obj_value
        else:
            raise ValueError("x must be 'nfe' or 'iter'")
        if error:
            if r.optimum is None:
                raise ValueError("error=True needs a problem with a known optimum")
            ys = np.abs(ys - r.optimum)
        label = labels[k] if labels else r.algorithm
        if x == "nfe":
            # the best value holds from the evaluation that found it onward
            ax.step(xs, ys, where="post", label=label)
        else:
            ax.plot(xs, ys, label=label)
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


PAPER_SIZE = (3.5, 2.6)
FONT = 8


def _axes(ax):
    if ax is not None:
        return ax
    return _plt().figure(figsize=PAPER_SIZE, layout="constrained").gca()


def _finish(ax, xlabel, ylabel, values, log, legend):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    pos = finite[finite > 0]
    if log is None:
        log = pos.size > 0 and pos.max() / pos.min() > 10
    if log:
        # values of 0 (an exact optimum) are drawn at the bottom edge
        ax.set_yscale("log", nonpositive="clip")
    ax.set_xlabel(xlabel, fontsize=FONT)
    ax.set_ylabel(ylabel, fontsize=FONT)
    ax.tick_params(labelsize=FONT)
    if legend:
        ax.legend(fontsize=FONT - 1, frameon=False)
    ax.grid(True, alpha=0.3)
    return ax


def band(x, curves, *, center="median", log=None, ax=None, xlabel="Function evaluations",
         ylabel="Error", title=None):
    """Convergence over several runs: ``curves`` maps a label to an array (runs, len(x)).

    Draws the median (or the mean, with ``center="mean"``) of the runs and a band from the
    25th to the 75th percentile. ``log`` defaults to True when the values are positive and
    span more than a factor of 10 (zeros are drawn at the bottom). Returns the Axes.
    """
    if center not in ("median", "mean"):
        raise ValueError("center must be 'median' or 'mean'")
    ax = _axes(ax)
    shown = []
    for label, y in curves.items():
        y = np.asarray(y, dtype=float)
        mid = np.median(y, axis=0) if center == "median" else np.mean(y, axis=0)
        lo, hi = np.percentile(y, [25, 75], axis=0)
        (line,) = ax.plot(x, mid, label=label, lw=1)
        if len(y) > 1:
            ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
        shown += [mid, lo, hi]
    if title:
        ax.set_title(title, fontsize=FONT)
    return _finish(ax, xlabel, ylabel, np.concatenate(shown), log, len(curves) > 1)


def boxes(samples, *, log=None, ax=None, ylabel="Final error", title=None):
    """Box plots of final values: ``samples`` maps a label to the values of its runs."""
    ax = _axes(ax)
    data = [np.asarray(v, dtype=float) for v in samples.values()]
    ax.boxplot(data, widths=0.6, flierprops={"markersize": 3})
    ax.set_xticks(range(1, len(data) + 1), list(samples), rotation=30 if len(data) > 4 else 0)
    if title:
        ax.set_title(title, fontsize=FONT)
    return _finish(ax, "", ylabel, np.concatenate(data), log, False)


def rank_bars(ranks, *, ax=None, ylabel="Average Friedman rank"):
    """Bar chart of average ranks (lower is better): ``ranks`` maps a label to its rank."""
    ax = _axes(ax)
    ax.bar(list(ranks), list(ranks.values()), color="0.45", width=0.6)
    ax.tick_params(axis="x", rotation=30 if len(ranks) > 4 else 0)
    ax.set_ylim(0, max(len(ranks), *ranks.values()) + 0.2)
    _finish(ax, "", ylabel, [], False, False)
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def front(*fronts, true_front=None, labels=None, kind=None, ax=None, objectives=None):
    """Objective vectors of one or more fronts (arrays of shape (n, m), a ``Front`` or a
    ``Result``), drawn as given, so maximized objectives are not flipped.

    Two objectives give a 2D scatter plot and three a 3D scatter plot. More objectives,
    or ``kind="parallel"``, give parallel coordinates: one line per point across the
    objectives. ``true_front`` (for example from ``ypea.benchmarks.multi.pareto_front``) is
    drawn in gray behind the fronts. ``objectives`` names the axes (default f1, f2, ...).
    A 3D plot needs ``ax`` to be a 3D axes (``projection="3d"``) or None. Returns the Axes.
    """
    plt = _plt()
    fronts = [getattr(getattr(f, "front", f), "obj_values", f) for f in fronts]
    sets = [np.atleast_2d(np.asarray(f, dtype=float)) for f in fronts]
    if true_front is not None:
        true_front = np.atleast_2d(np.asarray(true_front, dtype=float))
    every = sets + ([true_front] if true_front is not None else [])
    if not every:
        raise ValueError("front needs at least one array")
    m = every[0].shape[1]
    if any(f.ndim != 2 or f.shape[1] != m for f in every):
        raise ValueError("all fronts must have shape (n, m) with the same m")
    if kind is None:
        kind = "parallel" if m > 3 else "scatter"
    if kind not in ("scatter", "parallel"):
        raise ValueError("kind must be 'scatter' or 'parallel'")
    if kind == "scatter" and m not in (2, 3):
        raise ValueError("a scatter plot needs 2 or 3 objectives; use kind='parallel'")
    names = list(objectives) if objectives is not None else [f"$f_{{{i + 1}}}$" for i in range(m)]
    labels = labels or ([f"Front {k + 1}" for k in range(len(sets))] if len(sets) > 1 else [None])
    if ax is None:
        fig = plt.figure(figsize=(5, 4), layout="constrained")
        ax = fig.add_subplot(projection="3d" if kind == "scatter" and m == 3 else None)
    elif kind == "scatter" and m == 3 and getattr(ax, "name", "") != "3d":
        raise ValueError("a 3D front needs an axes made with projection='3d'")
    if kind == "parallel":
        x = np.arange(1, m + 1)
        if true_front is not None:
            ax.plot(x, true_front.T, color="0.75", lw=0.5, zorder=1)
        for k, f in enumerate(sets):
            lines = ax.plot(x, f.T, color=f"C{k}", lw=0.8, alpha=0.6, zorder=2)
            if labels[k] is not None and lines:
                lines[0].set_label(labels[k])
        ax.set_xticks(x, names)
        ax.set_ylabel("Objective value")
        ax.grid(True, axis="x", alpha=0.3)
    else:
        if true_front is not None:
            ax.scatter(*true_front.T, s=2, color="0.7", label="True front", zorder=1)
        for k, f in enumerate(sets):
            ax.scatter(*f.T, s=12, color=f"C{k}", label=labels[k], zorder=2)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        if m == 3:
            ax.set_zlabel(names[2])
        else:
            ax.grid(True, alpha=0.3)
    if any(lab is not None for lab in labels) or (true_front is not None and kind == "scatter"):
        ax.legend()
    return ax
