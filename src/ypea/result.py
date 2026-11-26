"""Result of a run and its history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .population import Individual


@dataclass
class History:
    """Per iteration records, plus the exact trace of improvements.

    ``iteration``, ``nfe``, ``best_obj_value``, ``best_violation`` and ``time`` have one
    entry per iteration (iteration 0 is the initialization). ``trace_nfe``,
    ``trace_obj_value`` and ``trace_violation`` have one entry per improvement of the best.
    ``trace_feasible_nfe`` and ``trace_feasible_obj_value`` have one entry per improvement
    of the best feasible solution.

    With several objectives, ``best_obj_value`` is NaN, ``best_violation`` is the smallest
    violation in the front, and ``archive_size`` gives the size of the front, all per
    iteration. The trace then logs the front exactly: one entry per solution that entered
    it, with ``trace_obj_value`` of shape (entries, n_obj), and ``trace_removed_nfe`` the
    NFE at which that solution left the front (-1 if it is still there). ``front_at(nfe)``
    rebuilds the front at any number of evaluations, for example to compute an indicator.
    """

    iteration: np.ndarray
    nfe: np.ndarray
    best_obj_value: np.ndarray
    best_violation: np.ndarray
    time: np.ndarray
    trace_nfe: np.ndarray
    trace_obj_value: np.ndarray
    trace_violation: np.ndarray
    trace_feasible_nfe: np.ndarray | None = None
    trace_feasible_obj_value: np.ndarray | None = None
    archive_size: np.ndarray | None = None
    trace_removed_nfe: np.ndarray | None = None

    def at_nfe(self, points) -> np.ndarray:
        """Best objective value found within each number of evaluations in ``points``.

        Returns NaN for points before the first evaluation. One objective only.
        """
        if self.trace_removed_nfe is not None:
            raise ValueError("a multi-objective run has no best value: use front_at(nfe)")
        points = np.asarray(points)
        idx = np.searchsorted(self.trace_nfe, points, side="right") - 1
        out = np.where(idx >= 0, self.trace_obj_value[np.maximum(idx, 0)], np.nan)
        return out

    def front_at(self, nfe) -> np.ndarray:
        """Several objectives: a boolean mask of the trace entries that formed the front
        after ``nfe`` evaluations. ``trace_obj_value[mask]`` are their objective values."""
        if self.trace_removed_nfe is None:
            raise ValueError("front_at needs a multi-objective run")
        gone = self.trace_removed_nfe
        return (self.trace_nfe <= nfe) & ((gone < 0) | (gone > nfe))


@dataclass
class Front:
    """The non-dominated solutions of a multi-objective run (``Result.front``).

    ``positions`` has shape (k, code_size), ``obj_values`` (k, n_obj) and ``violations``
    (k,). ``solutions`` holds the k decoded solutions. The rows are sorted by the first
    objective (then the second, and so on), best first. ``front[i]`` gives an
    ``Individual`` whose ``obj_value`` is an array.
    """

    positions: np.ndarray
    obj_values: np.ndarray
    violations: np.ndarray
    solutions: list

    def __len__(self):
        return len(self.violations)

    def __getitem__(self, i) -> Individual:
        return Individual(self.positions[i], self.solutions[i], self.obj_values[i], float(self.violations[i]))

    @property
    def feasible(self) -> np.ndarray:
        """True for each feasible solution (all of them, once any feasible solution was found)."""
        return self.violations <= 0


@dataclass
class Evaluations:
    """Every evaluated solution of a run (``Result.evaluations``), in evaluation order.

    ``nfe`` is the evaluation number of each row (1, 2, ...), ``positions`` the codes,
    ``obj_values`` the objective values (shape (n,), or (n, n_obj) with several
    objectives) and ``violations`` the total constraint violations. Non-finite objective
    values are stored as the worst value, as the run saw them. Index it like an array to
    filter or sort: ``ev[ev.feasible]``, ``ev[np.argsort(ev.obj_values)]``. Decode rows
    with ``problem.decode(ev.positions)``.
    """

    nfe: np.ndarray
    positions: np.ndarray
    obj_values: np.ndarray
    violations: np.ndarray

    def __len__(self):
        return len(self.nfe)

    def __getitem__(self, idx) -> Evaluations:
        idx = [idx] if np.isscalar(idx) else idx
        return Evaluations(self.nfe[idx], self.positions[idx], self.obj_values[idx], self.violations[idx])

    @property
    def feasible(self) -> np.ndarray:
        return self.violations <= 0

    def to_dict(self) -> dict:
        return _plain({"nfe": self.nfe, "positions": self.positions, "obj_values": self.obj_values,
                       "violations": self.violations})


def _individual(ind):
    if ind is None:
        return None
    return {"obj_value": ind.obj_value, "violation": ind.violation, "feasible": ind.feasible,
            "position": ind.position,
            "solution": vars(ind.solution) if hasattr(ind.solution, "__dict__") else ind.solution}


@dataclass
class Result:
    """What ``Algorithm.solve`` returns.

    ``best`` is the best solution by the problem's comparison rules (with
    ``constraints="penalty"`` it may be infeasible). ``best_feasible`` is the best feasible
    solution by objective value alone, or None when no feasible solution was found. With
    the default feasibility rules it equals ``best`` once a feasible solution exists.

    With several objectives, ``best`` and ``best_feasible`` are None, and ``front`` holds
    the non-dominated solutions (a ``Front``). ``evaluations`` holds every evaluated
    solution when the run was started with ``keep_evaluations=True``.
    """

    best: Individual | None
    history: History
    algorithm: str
    params: dict
    problem: str
    seed: Any
    nfe: int
    iterations: int
    run_time: float
    stop_reason: str
    optimum: float | None = None
    extra: dict = field(default_factory=dict)
    front: Front | None = None
    best_feasible: Individual | None = None
    evaluations: Evaluations | None = None

    @property
    def error(self) -> float | None:
        """Distance of the best objective value from the known optimum, if there is one."""
        if self.optimum is None or self.best is None:
            return None
        return abs(self.best.obj_value - self.optimum)

    def to_dict(self) -> dict:
        """Plain Python data (JSON ready): settings, seed, best solution (or the front),
        history and trace, and the evaluations when they were kept."""
        h = self.history
        d = {
            "algorithm": self.algorithm, "problem": self.problem, "params": self.params,
            "seed": _seed(self.seed), "nfe": self.nfe, "iterations": self.iterations,
            "run_time": self.run_time, "stop_reason": self.stop_reason, "optimum": self.optimum,
            "error": self.error,
            "best": _individual(self.best),
            "history": {"iteration": h.iteration, "nfe": h.nfe, "best_obj_value": h.best_obj_value,
                        "best_violation": h.best_violation, "time": h.time},
            "trace": {"nfe": h.trace_nfe, "obj_value": h.trace_obj_value, "violation": h.trace_violation},
            "extra": self.extra,
        }
        if self.front is None:
            d["best_feasible"] = _individual(self.best_feasible)
            d["feasible_trace"] = {"nfe": h.trace_feasible_nfe, "obj_value": h.trace_feasible_obj_value}
        else:
            f = self.front
            d["front"] = {"obj_values": f.obj_values, "violations": f.violations, "positions": f.positions,
                          "solutions": [vars(s) if hasattr(s, "__dict__") else s for s in f.solutions]}
            d["history"]["archive_size"] = h.archive_size
            d["trace"]["removed_nfe"] = h.trace_removed_nfe
        if self.evaluations is not None:
            d["evaluations"] = self.evaluations.to_dict()
        return _plain(d)

    def summary(self) -> str:
        if self.front is not None:
            f = self.front
            return "\n".join([
                f"{self.algorithm} on {self.problem}",
                f"  front: {len(f)} solutions, {int(f.feasible.sum())} feasible",
                "  objective ranges: " + ", ".join(f"[{lo:.6g}, {hi:.6g}]" for lo, hi in
                                                     zip(f.obj_values.min(axis=0), f.obj_values.max(axis=0),
                                                         strict=True)),
                f"  evaluations: {self.nfe}, iterations: {self.iterations}, "
                f"time: {self.run_time:.3g} s, stopped by: {self.stop_reason}",
            ])
        lines = [
            f"{self.algorithm} on {self.problem}",
            f"  best objective value: {self.best.obj_value:.6g}",
            f"  feasible: {self.best.feasible} (violation {self.best.violation:.3g})",
            f"  evaluations: {self.nfe}, iterations: {self.iterations}, "
            f"time: {self.run_time:.3g} s, stopped by: {self.stop_reason}",
        ]
        if self.optimum is not None:
            lines.insert(2, f"  error: {self.error:.6g}")
        if not self.best.feasible:
            b = self.best_feasible
            lines.insert(-1, "  best feasible: none found" if b is None else
                         f"  best feasible objective value: {b.obj_value:.6g}")
        return "\n".join(lines)


def _seed(seed):
    """A JSON form of a run's seed: an int, None, {"studio": n}, or a description."""
    from .random import StudioRng

    if seed is None or isinstance(seed, (int, np.integer)):
        return None if seed is None else int(seed)
    if isinstance(seed, StudioRng):
        return {"studio": seed.seed}
    return repr(seed)


def _plain(obj):
    """Turn NumPy values, arrays, tuples and nested containers into plain JSON data."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)  # JSON has no inf or NaN
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    return repr(obj)  # for example an operator object among the settings
