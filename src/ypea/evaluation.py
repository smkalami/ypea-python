"""Evaluation of codes: decode, call the objective, count evaluations, track the best."""

from __future__ import annotations

import numpy as np

from .comparison import _equal
from .population import Population
from .problem import Problem, _scalar, parse_batch, parse_result


class StopRun(BaseException):  # noqa: N818 (a signal, not an error)
    """Raised inside a run to end it early with ``reason`` as the stop reason (used by ask and tell)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def parse_values(raw, n, vectorized, n_obj=1, eq=None):
    """Turn the objective values of a batch of n solutions into (F, V, constrained).

    ``raw`` is what a vectorized objective returns for the batch, or else a sequence with
    one return value per solution. F has shape (n,), or (n, n_obj) with several objectives.
    ``eq`` holds the equality tolerances of the problem.
    """
    if vectorized:
        return (*parse_batch(raw, n, n_obj, eq), isinstance(raw, tuple))
    raw = list(raw)
    if len(raw) != n:
        raise ValueError(f"expected {n} objective values, got {len(raw)}")
    F, V = np.empty(n if n_obj == 1 else (n, n_obj)), np.zeros(n)
    constrained = False
    for k, r in enumerate(raw):
        if isinstance(r, tuple):
            constrained = True
            F[k], V[k] = parse_result(r, n_obj, eq)
        elif n_obj == 1:
            F[k] = _scalar(r)
        else:
            F[k] = parse_result(r, n_obj)[0]
    return F, V, constrained


class Evaluator:
    """Evaluates batches of codes for one run.

    Contracts:

    * The returned Population always has as many rows as ``X``. Rows past the
      ``max_nfe`` budget are not evaluated: they get the worst value and infinite
      violation, so they never win a comparison. ``stop_reason`` is then set.
    * With one objective, the best solution so far is updated after every single
      evaluation, and every improvement is recorded in ``trace`` as (nfe, objective value,
      violation). The best feasible solution is tracked the same way, in its own trace.
    * With several objectives, ``archive`` holds every non-dominated solution found so
      far, updated as if after every single evaluation (see ``_update_archive``). Each
      solution that enters it is recorded in ``trace`` as (nfe, objective values,
      violation), and ``trace_removed`` holds the NFE at which it left the archive, or -1.
    * With ``keep_evaluations``, ``log`` keeps every evaluated batch (codes, values and
      violations). Rows past the budget are not kept.
    """

    def __init__(self, problem: Problem, max_nfe=None, target=None, executor=None, *,
                 max_archive=None, keep_evaluations=False):
        self.problem = problem
        self.cmp = problem.comparator
        self.n_obj = problem.n_obj
        self.max_nfe = max_nfe
        self.target = target
        self.executor = executor
        self.max_archive = max_archive
        self.nfe = 0
        self.best = None  # Population of length 1 (one objective)
        self.best_feasible = None  # Population of length 1 (one objective)
        self.archive = None  # Population (several objectives)
        self._archive_ids = np.zeros(0, dtype=int)  # trace index of each archive member
        self.trace_nfe, self.trace_f, self.trace_v = [], [], []
        self.trace_removed = []  # several objectives
        self.feasible_trace_nfe, self.feasible_trace_f = [], []
        self.log = [] if keep_evaluations else None
        self.stop_reason = None
        self.constrained = False  # set when the objective returns (f, g)

    def __call__(self, X) -> Population:
        X = np.clip(np.atleast_2d(np.asarray(X, dtype=float)), 0.0, 1.0)
        n = len(X)
        single = self.n_obj == 1
        F = np.full(n if single else (n, self.n_obj), self.cmp.worst_value)
        V = np.full(n, np.inf)
        m = n if self.max_nfe is None else max(0, min(n, self.max_nfe - self.nfe))
        if m:
            f, v = self._evaluate(X[:m] if m < n else X)
            # NaN, inf and -inf all count as the worst value (-inf would otherwise win a minimization)
            if single:
                F[:m] = np.where(np.isfinite(f), f, self.cmp.worst_value)
            else:  # a row with any non-finite objective is the worst in every objective
                F[:m] = np.where(np.isfinite(f).all(axis=1, keepdims=True), f, self.cmp.worst_value)
            V[:m] = np.where(np.isnan(v), np.inf, v)
            if single:
                self._update_best(X[:m], F[:m], V[:m])
                self._update_feasible(X[:m], F[:m], V[:m])
            else:
                self._update_archive(X[:m], F[:m], V[:m])
            if self.log is not None:
                self.log.append((X[:m].copy(), F[:m].copy(), V[:m].copy()))
            self.nfe += m
        if self.max_nfe is not None and self.nfe >= self.max_nfe and self.stop_reason is None:
            self.stop_reason = "max_nfe"
        return Population(X, F, V)

    def _evaluate(self, X):
        p = self.problem
        sols = p.decode(X)
        if p.vectorized:
            raw = p.objective(sols)
        else:
            raw = list(map(p.objective, sols)) if self.executor is None else list(self.executor.map(p.objective, sols))
        F, V, constrained = parse_values(raw, len(X), p.vectorized, p.n_obj, p.equality_tolerance)
        self.constrained |= constrained
        return F, V

    def _update_best(self, X, F, V):
        p, s = self.cmp.keys(F[:, None], V)
        if self.best is None:
            cand = range(len(p))
        else:
            bp, bs = self._best_keys
            cand = np.flatnonzero((p < bp) | ((p == bp) & (s < bs)))
            if not len(cand):
                return  # the best did not change, so neither did the target check
        # walk the candidates in order, so each improvement gets its exact NFE
        last = None
        for i in cand:
            pi, si = float(p[i]), float(s[i])
            if (last is None and self.best is None) or pi < bp or (pi == bp and si < bs):
                bp, bs, last = pi, si, int(i)
                self.trace_nfe.append(self.nfe + last + 1)
                self.trace_f.append(float(F[last]))
                self.trace_v.append(float(V[last]))
        self._best_keys = (bp, bs)
        self.best = Population(X[last:last + 1].copy(), F[last:last + 1, None].copy(), V[last:last + 1].copy())
        if (self.target is not None and self.stop_reason is None
                and self.cmp.reached(self.best.F, self.best.V, self.target)[0]):
            self.stop_reason = "target"

    def _update_feasible(self, X, F, V):
        """The best feasible solution by objective value alone (the first of equal ones)."""
        f = self.cmp.sign * F
        ok = V <= 0
        if self.best_feasible is None:
            cand = np.flatnonzero(ok)
        else:
            bf = self._feasible_key
            cand = np.flatnonzero(ok & (f < bf))
        last = None
        for i in cand:
            if (last is None and self.best_feasible is None) or f[i] < bf:
                bf, last = float(f[i]), int(i)
                self.feasible_trace_nfe.append(self.nfe + last + 1)
                self.feasible_trace_f.append(float(F[last]))
        if last is not None:
            self._feasible_key = bf
            self.best_feasible = Population(X[last:last + 1].copy(), F[last:last + 1, None].copy(),
                                            V[last:last + 1].copy())

    def _dominates(self, F1, V1, F2, V2):
        # the archive always follows constrained domination, whatever the run's constraint handling
        return self.cmp.dominates(F1, V1, F2, V2, rule="feasibility")

    def _update_archive(self, X, F, V):
        """Add a batch to the archive, with the same result and trace as adding its rows one
        at a time. A row enters when no archive member and no earlier row of the batch
        dominates it or has the same objective values and violation. It removes the members
        it dominates. Then, with ``max_archive``, the most crowded member is removed one at
        a time (the crowding distance is computed again after each removal) until the
        archive fits. So inside a batch the archive may hold more than ``max_archive``."""
        n = len(X)
        A = self.archive

        def blocks(F1, V1, F2, V2):  # blocks[i, j]: row i dominates or equals row j
            same = _equal(F1, F2) & (V1 == V2)
            return self._dominates(F1, V1, F2, V2) | same

        blocked = (blocks(F[:, None], V[:, None], F[None], V[None]) & np.triu(np.ones((n, n), bool), 1)).any(axis=0)
        if A is not None and len(A):
            blocked |= blocks(A.F[:, None], A.V[:, None], F[None], V[None]).any(axis=0)
        new = np.flatnonzero(~blocked)
        if not len(new):
            return
        first_id = len(self.trace_nfe)
        nfe_new = self.nfe + new + 1
        self.trace_nfe += nfe_new.tolist()
        self.trace_f += [row for row in F[new].copy()]
        self.trace_v += V[new].tolist()
        self.trace_removed += [-1] * len(new)

        def removed_by(kill):  # kill[i, j]: entrant i removes member j. NFE of the first such entrant, or 0
            hit = kill.any(axis=0)
            return np.where(hit, nfe_new[np.argmax(kill, axis=0)], 0)

        Fn, Vn = F[new], V[new]
        # entrants removed by later entrants
        later = np.tril(np.ones((len(new), len(new)), bool), -1)
        kill = self._dominates(Fn[:, None], Vn[:, None], Fn[None], Vn[None]) & later
        gone_new = removed_by(kill)
        keep_new = gone_new == 0
        for k in np.flatnonzero(~keep_new):
            self.trace_removed[first_id + k] = int(gone_new[k])
        X_new, F_new, V_new, ids_new = X[new][keep_new], Fn[keep_new], Vn[keep_new], first_id + np.flatnonzero(keep_new)
        if A is not None and len(A):
            gone = removed_by(self._dominates(Fn[:, None], Vn[:, None], A.F[None], A.V[None]))
            for j in np.flatnonzero(gone):
                self.trace_removed[self._archive_ids[j]] = int(gone[j])
            keep = gone == 0
            X_new, F_new, V_new = (np.concatenate([A.X[keep], X_new]), np.concatenate([A.F[keep], F_new]),
                                   np.concatenate([A.V[keep], V_new]))
            ids_new = np.concatenate([self._archive_ids[keep], ids_new])
        if self.max_archive is not None and len(V_new) > self.max_archive:
            keep = np.ones(len(V_new), bool)
            for _ in range(len(V_new) - self.max_archive):
                idx = np.flatnonzero(keep)
                j = idx[np.argmin(self.cmp.crowding_distance(F_new[idx]))]
                keep[j] = False
                self.trace_removed[ids_new[j]] = self.nfe + n
            X_new, F_new, V_new, ids_new = X_new[keep], F_new[keep], V_new[keep], ids_new[keep]
        self.archive = Population(X_new, F_new, V_new)
        self._archive_ids = ids_new
