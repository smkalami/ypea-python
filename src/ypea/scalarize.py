"""Turn a multi-objective problem into a single objective one."""

from __future__ import annotations

import numpy as np

from .problem import Problem


class _Scalarized:
    """The objective of a scalarized problem (a class, so the problem can be pickled)."""

    def __init__(self, objective, sign, weights, method, ideal, vectorized):
        self.objective, self.sign, self.weights = objective, sign, weights
        self.method, self.ideal, self.vectorized = method, ideal, vectorized

    def __call__(self, sol):
        out = self.objective(sol)
        rest = out[1:] if isinstance(out, tuple) else ()
        f = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)
        m = len(self.weights)
        f = f.reshape(-1, m) if self.vectorized else f.reshape(m)
        if self.method == "weighted_sum":
            s = (self.weights * self.sign * f).sum(axis=-1)
        else:
            s = (self.weights * np.abs(f - self.ideal)).max(axis=-1)
        s = s if self.vectorized else float(s)
        return (s, *rest) if rest else s


def scalarize(problem, weights, method="weighted_sum", ideal=None) -> Problem:
    """A single objective problem from a multi-objective one, to minimize.

    Parameters
    ----------
    problem : Problem
        A problem with several objectives.
    weights : list of float
        One non negative weight per objective, not all zero.
    method : "weighted_sum" or "tchebycheff"
        ``"weighted_sum"`` minimizes ``sum(w_i * s_i * f_i)``, with ``s_i = 1`` for an
        objective to minimize and -1 for one to maximize. It finds only points on the
        convex parts of the front. ``"tchebycheff"`` minimizes ``max(w_i * |f_i - z_i|)``
        (Zhang and Li 2007), which can reach every point of the front.
    ideal : list of float
        The ideal point ``z`` for Tchebycheff, in the units of the objectives: a value at
        least as good as the best possible value of each objective. Required for
        ``"tchebycheff"``.

    The new problem has the same variables, constraints and settings. Its objective calls
    the original one and combines the values, so constraint values pass through.
    """
    if not isinstance(problem, Problem):
        raise TypeError("problem must be a ypea.Problem")
    if problem.n_obj < 2:
        raise ValueError("the problem has one objective already")
    if problem.objective is None:
        raise ValueError("the problem has no objective to scalarize")
    w = np.asarray(weights, dtype=float)
    if w.shape != (problem.n_obj,) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError(f"weights must be {problem.n_obj} non negative numbers, not all zero")
    if method not in ("weighted_sum", "tchebycheff"):
        raise ValueError("method must be 'weighted_sum' or 'tchebycheff'")
    z = None
    if method == "tchebycheff":
        if ideal is None:
            raise ValueError("tchebycheff needs the ideal point: ideal=[z1, z2, ...]")
        z = np.asarray(ideal, dtype=float)
        if z.shape != (problem.n_obj,) or not np.all(np.isfinite(z)):
            raise ValueError(f"ideal must be {problem.n_obj} finite numbers")
    objective = _Scalarized(problem.objective, problem.comparator.sign, w, method, z, problem.vectorized)
    return Problem(problem.vars, objective, "min", vectorized=problem.vectorized,
                   constraints=problem.constraints, penalty=problem.penalty,
                   equality_atol=problem.equality_atol, equality_rtol=problem.equality_rtol,
                   equality_scale=problem.equality_scale, name=f"{problem.name} ({method})",
                   info={**problem.info, "scalarized": {"weights": w.tolist(), "method": method,
                                                        "ideal": None if z is None else z.tolist()}})
