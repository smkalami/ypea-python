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
    """

    iteration: np.ndarray
    nfe: np.ndarray
    best_obj_value: np.ndarray
    best_violation: np.ndarray
    time: np.ndarray
    trace_nfe: np.ndarray
    trace_obj_value: np.ndarray
    trace_violation: np.ndarray

    def at_nfe(self, points) -> np.ndarray:
        """Best objective value found within each number of evaluations in ``points``.

        Returns NaN for points before the first evaluation.
        """
        points = np.asarray(points)
        idx = np.searchsorted(self.trace_nfe, points, side="right") - 1
        out = np.where(idx >= 0, self.trace_obj_value[np.maximum(idx, 0)], np.nan)
        return out


@dataclass
class Result:
    """What ``Algorithm.solve`` returns."""

    best: Individual
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

    @property
    def error(self) -> float | None:
        """Distance of the best objective value from the known optimum, if there is one."""
        if self.optimum is None:
            return None
        return abs(self.best.obj_value - self.optimum)

    def summary(self) -> str:
        lines = [
            f"{self.algorithm} on {self.problem}",
            f"  best objective value: {self.best.obj_value:.6g}",
            f"  feasible: {self.best.feasible} (violation {self.best.violation:.3g})",
            f"  evaluations: {self.nfe}, iterations: {self.iterations}, "
            f"time: {self.run_time:.3g} s, stopped by: {self.stop_reason}",
        ]
        if self.optimum is not None:
            lines.insert(2, f"  error: {self.error:.6g}")
        return "\n".join(lines)
