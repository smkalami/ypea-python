"""Comparison of solutions. This is the only module that knows about min or max and constraints.

Solutions are compared by two keys, both to be minimized, in lexicographic order:

* ``"feasibility"`` (Deb, 2000): first the total constraint violation, then the objective.
  A feasible solution beats an infeasible one, two infeasible ones are compared by
  violation, and two feasible ones by objective.
* ``"penalty"``: first the penalized objective ``f + penalty * violation``, then the violation.
"""

from __future__ import annotations

import numpy as np

_SENSES = {
    "min": 1, "minimize": 1, "minimization": 1,
    "max": -1, "maximize": -1, "maximization": -1,
}


def parse_sense(sense: str) -> str:
    try:
        return "min" if _SENSES[str(sense).lower()] == 1 else "max"
    except KeyError:
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}") from None


class Comparator:
    def __init__(self, sense="min", constraint_handling="feasibility", penalty=1e6):
        self.sense = parse_sense(sense)
        self.sign = 1.0 if self.sense == "min" else -1.0
        if constraint_handling not in ("feasibility", "penalty"):
            raise ValueError("constraints must be 'feasibility' or 'penalty'")
        self.constraint_handling = constraint_handling
        self.penalty = float(penalty)
        if not self.penalty >= 0:
            raise ValueError("penalty must be non negative")

    @property
    def worst_value(self) -> float:
        return self.sign * np.inf

    def keys(self, F, V):
        """(primary, secondary) keys to minimize. F has shape (n, 1), V has shape (n,)."""
        f = self.sign * np.asarray(F, dtype=float)[..., 0]
        V = np.asarray(V, dtype=float)
        if self.constraint_handling == "feasibility":
            return V, f
        with np.errstate(invalid="ignore"):
            p = f + self.penalty * V
        return np.where(np.isnan(p), np.inf, p), V

    def better(self, F1, V1, F2, V2):
        """Element wise: is solution 1 strictly better than solution 2? Broadcasts."""
        p1, s1 = self.keys(F1, V1)
        p2, s2 = self.keys(F2, V2)
        return (p1 < p2) | ((p1 == p2) & (s1 < s2))

    def argsort(self, F, V):
        """Indices from best to worst. Stable, so ties keep their order."""
        p, s = self.keys(F, V)
        return np.lexsort((s, p))

    def ranks(self, F, V):
        """Rank of each solution, 0 for the best. Tied solutions get the same (lowest) rank."""
        p, s = self.keys(F, V)
        order = np.lexsort((s, p))
        ranks = np.empty(len(order), dtype=float)
        # equal keys share the rank of the first one
        ps, ss = p[order], s[order]
        same = np.r_[False, (ps[1:] == ps[:-1]) & (ss[1:] == ss[:-1])]
        first = np.maximum.accumulate(np.where(same, 0, np.arange(len(order))))
        ranks[order] = first
        return ranks

    def reached(self, F, V, target) -> np.ndarray:
        f = self.sign * np.asarray(F, dtype=float)[..., 0]
        return (np.asarray(V) <= 0) & (f <= self.sign * target)
