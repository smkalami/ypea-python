"""Comparison of solutions. This is the only module that knows about min or max and constraints.

Solutions are compared by two keys, both to be minimized, in lexicographic order:

* ``"feasibility"`` (Deb, 2000): first the total constraint violation, then the objective.
  A feasible solution beats an infeasible one, two infeasible ones are compared by
  violation, and two feasible ones by objective.
* ``"penalty"``: first the penalized objective ``f + penalty * violation``, then the violation.

With several objectives, solutions are compared by dominance (``Comparator.dominates``),
and sets of solutions are ordered by non-dominated sorting and the crowding distance of
Deb, Pratap, Agarwal and Meyarivan (2002).
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


def _pareto(a, b):
    """Element wise over the last axis: does a Pareto dominate b (both to be minimized)?"""
    # one objective at a time: a reduction over a short last axis is many times slower
    a, b = np.broadcast_arrays(a, b)
    le, lt = a[..., 0] <= b[..., 0], a[..., 0] < b[..., 0]
    for k in range(1, a.shape[-1]):
        le &= a[..., k] <= b[..., k]
        lt |= a[..., k] < b[..., k]
    return le & lt


def _equal(a, b):
    """Element wise over the last axis: are all entries equal?"""
    a, b = np.broadcast_arrays(a, b)
    eq = a[..., 0] == b[..., 0]
    for k in range(1, a.shape[-1]):
        eq &= a[..., k] == b[..., k]
    return eq


class Comparator:
    """Compares solutions by the senses of the objectives and the constraint handling.

    ``sense`` is "min" or "max" for one objective, or a list with one of them per
    objective. With one objective, ``sense`` is a string and ``sign`` is 1.0 or -1.0.
    With several, ``sense`` is a list and ``sign`` an array with one entry per objective.
    """

    def __init__(self, sense="min", constraint_handling="feasibility", penalty=1e6):
        senses = [sense] if isinstance(sense, str) else list(sense)
        if not senses:
            raise ValueError("sense must be 'min', 'max' or a non empty list of them")
        self.n_obj = len(senses)
        if self.n_obj == 1:
            self.sense = parse_sense(senses[0])
            self.sign = 1.0 if self.sense == "min" else -1.0
        else:
            self.sense = [parse_sense(s) for s in senses]
            self.sign = np.array([1.0 if s == "min" else -1.0 for s in self.sense])
        if constraint_handling not in ("feasibility", "penalty"):
            raise ValueError("constraints must be 'feasibility' or 'penalty'")
        self.constraint_handling = constraint_handling
        self.penalty = float(penalty)
        if not self.penalty >= 0:
            raise ValueError("penalty must be non negative")

    @property
    def worst_value(self):
        """The worst objective value: a number, or an array with one entry per objective."""
        return self.sign * np.inf

    def _single(self, what):
        if self.n_obj > 1:
            raise ValueError(f"{what} needs a single objective. With several objectives use "
                             "dominates, nondominated_sort and crowding_distance")

    def keys(self, F, V):
        """(primary, secondary) keys to minimize. F has shape (n, 1), V has shape (n,)."""
        self._single("keys")
        f = self.sign * np.asarray(F, dtype=float)[..., 0]
        V = np.asarray(V, dtype=float)
        if self.constraint_handling == "feasibility":
            return V, f
        with np.errstate(invalid="ignore"):
            p = f + self.penalty * V
        return np.where(np.isnan(p), np.inf, p), V

    def better(self, F1, V1, F2, V2):
        """Element wise: is solution 1 strictly better than solution 2? Broadcasts.

        With several objectives this is ``dominates``."""
        if self.n_obj > 1:
            return self.dominates(F1, V1, F2, V2)
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
        same = np.zeros(len(order), bool)
        same[1:] = (ps[1:] == ps[:-1]) & (ss[1:] == ss[:-1])
        first = np.maximum.accumulate(np.where(same, 0, np.arange(len(order))))
        ranks[order] = first
        return ranks

    def reached(self, F, V, target) -> np.ndarray:
        self._single("target")
        f = self.sign * np.asarray(F, dtype=float)[..., 0]
        return (np.asarray(V) <= 0) & (f <= self.sign * target)

    # several objectives (these work for one objective too)

    def dominates(self, F1, V1, F2, V2, rule=None):
        """Element wise: does solution 1 dominate solution 2? Broadcasts.

        ``F`` has shape (..., m) and ``V`` shape (...). ``rule`` is the constraint handling,
        "feasibility" or "penalty", by default the comparator's own.

        * ``"feasibility"`` (constrained domination, Deb et al. 2002): a feasible solution
          dominates an infeasible one, an infeasible one dominates another with a larger
          violation, and two feasible ones are compared by Pareto dominance. Two infeasible
          solutions with exactly the same violation are also compared by Pareto dominance
          (the paper leaves both non-dominated).
        * ``"penalty"``: Pareto dominance of the penalized objectives ``f + penalty * v``
          (each objective turned into minimization first). Equal penalized objectives are
          decided by the lower violation.

        With one objective both rules give the same answer as ``better``.
        """
        rule = rule or self.constraint_handling
        V1, V2 = np.asarray(V1, dtype=float), np.asarray(V2, dtype=float)
        a, b = self.sign * np.asarray(F1, dtype=float), self.sign * np.asarray(F2, dtype=float)
        if rule == "feasibility":
            return (V1 < V2) | ((V1 == V2) & _pareto(a, b))
        with np.errstate(invalid="ignore"):
            a, b = a + self.penalty * V1[..., None], b + self.penalty * V2[..., None]
        a, b = np.where(np.isnan(a), np.inf, a), np.where(np.isnan(b), np.inf, b)
        return _pareto(a, b) | (_equal(a, b) & (V1 < V2))

    def nondominated_sort(self, F, V, rule=None):
        """Fast non-dominated sorting (Deb et al. 2002) by ``dominates`` with ``rule``.

        Returns the front of each solution: 0 for the solutions that no other one
        dominates, 1 for those dominated only by solutions of front 0, and so on.
        Uses memory for an n by n matrix.
        """
        F, V = np.asarray(F, dtype=float), np.asarray(V, dtype=float)
        F = F.reshape(len(V), -1)
        dom = self.dominates(F[:, None], V[:, None], F[None], V[None], rule)  # dom[i, j]: i dominates j
        count = dom.sum(axis=0)
        ranks = np.full(len(V), -1)
        front, r = np.flatnonzero(count == 0), 0
        while len(front):
            ranks[front] = r
            count[front] = -1
            count -= dom[front].sum(axis=0)
            front, r = np.flatnonzero(count == 0), r + 1
        return ranks

    def crowding_distance(self, F, ranks=None):
        """Crowding distance of Deb et al. (2002), computed within each front (all
        solutions form one front when ``ranks`` is None). Larger means less crowded.

        For each objective, the front is sorted by that objective, the two boundary
        solutions get an infinite distance, and every other solution adds the gap between
        its two neighbors divided by the range of that objective in the front. A front of
        one or two solutions gets infinite distances, and an objective with no range adds
        nothing. The sort is stable, so of two equal values the first one comes first.
        """
        F = np.asarray(F, dtype=float)
        F = F.reshape(len(F), -1)
        d = np.zeros(len(F))
        groups = [np.arange(len(F))] if ranks is None else [np.flatnonzero(ranks == r) for r in np.unique(ranks)]
        for idx in groups:
            if len(idx) <= 2:
                d[idx] = np.inf
                continue
            f = F[idx]
            dist = np.zeros(len(idx))
            for k in range(f.shape[1]):
                o = np.argsort(f[:, k], kind="stable")
                v = f[o, k]
                span = v[-1] - v[0]
                if span > 0 and np.isfinite(span):
                    dist[o[1:-1]] += (v[2:] - v[:-2]) / span
                dist[o[0]] = dist[o[-1]] = np.inf
            d[idx] = dist
        return d
