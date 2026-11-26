"""Evaluation of codes: decode, call the objective, count evaluations, track the best."""

from __future__ import annotations

import numpy as np

from .population import Population
from .problem import Problem, _scalar, parse_batch, parse_result


class Evaluator:
    """Evaluates batches of codes for one run.

    Contracts:

    * The returned Population always has as many rows as ``X``. Rows past the
      ``max_nfe`` budget are not evaluated: they get the worst value and infinite
      violation, so they never win a comparison. ``stop_reason`` is then set.
    * The best solution so far is updated after every single evaluation, and every
      improvement is recorded in ``trace`` as (nfe, objective value, violation).
    """

    def __init__(self, problem: Problem, max_nfe=None, target=None, executor=None):
        self.problem = problem
        self.cmp = problem.comparator
        self.max_nfe = max_nfe
        self.target = target
        self.executor = executor
        self.nfe = 0
        self.best = None  # Population of length 1
        self.trace_nfe, self.trace_f, self.trace_v = [], [], []
        self.stop_reason = None
        self.constrained = False  # set when the objective returns (f, g)

    def __call__(self, X) -> Population:
        X = np.clip(np.atleast_2d(np.asarray(X, dtype=float)), 0.0, 1.0)
        n = len(X)
        F = np.full(n, self.cmp.worst_value)
        V = np.full(n, np.inf)
        m = n if self.max_nfe is None else max(0, min(n, self.max_nfe - self.nfe))
        if m:
            f, v = self._evaluate(X[:m] if m < n else X)
            # NaN, inf and -inf all count as the worst value (-inf would otherwise win a minimization)
            F[:m] = np.where(np.isfinite(f), f, self.cmp.worst_value)
            V[:m] = np.where(np.isnan(v), np.inf, v)
            self._update_best(X[:m], F[:m], V[:m])
            self.nfe += m
        if self.max_nfe is not None and self.nfe >= self.max_nfe and self.stop_reason is None:
            self.stop_reason = "max_nfe"
        return Population(X, F, V)

    def _evaluate(self, X):
        p = self.problem
        if p.vectorized:
            raw = p.objective(p.decode(X))
            self.constrained |= isinstance(raw, tuple)
            return parse_batch(raw, len(X))
        sols = p.decode(X)
        raw = list(map(p.objective, sols)) if self.executor is None else list(self.executor.map(p.objective, sols))
        F, V = np.empty(len(raw)), np.zeros(len(raw))
        for k, r in enumerate(raw):
            if isinstance(r, tuple):
                self.constrained = True
                F[k], V[k] = parse_result(r)
            else:
                F[k] = _scalar(r)
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
