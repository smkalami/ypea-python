"""Evaluation of codes: decode, call the objective, count evaluations, track the best."""

from __future__ import annotations

import numpy as np

from .population import Population
from .problem import Problem, parse_batch, parse_result


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

    def __call__(self, X) -> Population:
        X = np.clip(np.atleast_2d(np.asarray(X, dtype=float)), 0.0, 1.0)
        n = len(X)
        F = np.full(n, self.cmp.worst_value)
        V = np.full(n, np.inf)
        m = n if self.max_nfe is None else max(0, min(n, self.max_nfe - self.nfe))
        if m:
            F[:m], V[:m] = self._evaluate(X[:m])
            F[:m] = np.where(np.isnan(F[:m]), self.cmp.worst_value, F[:m])
            V[:m] = np.where(np.isnan(V[:m]), np.inf, V[:m])
            self._update_best(X[:m], F[:m], V[:m])
            self.nfe += m
        if self.max_nfe is not None and self.nfe >= self.max_nfe and self.stop_reason is None:
            self.stop_reason = "max_nfe"
        return Population(X, F, V)

    def _evaluate(self, X):
        p = self.problem
        if p.vectorized:
            return parse_batch(p.objective(p.decode(X)), len(X))
        sols = p.decode(X)
        if self.executor is None:
            out = [parse_result(p.objective(s)) for s in sols]
        else:
            out = [parse_result(r) for r in self.executor.map(p.objective, sols)]
        F, V = zip(*out, strict=True)
        return np.array(F), np.array(V)

    def _update_best(self, X, F, V):
        Fc, Vc = F[:, None], V
        # walk the batch in order, so each improvement gets its exact NFE
        cand = np.flatnonzero(self._better_than_best(Fc, Vc))
        for i in cand:
            if self.best is None or self.cmp.better(Fc[i:i + 1], Vc[i:i + 1], self.best.F, self.best.V)[0]:
                self.best = Population(X[i:i + 1].copy(), Fc[i:i + 1].copy(), Vc[i:i + 1].copy())
                self.trace_nfe.append(self.nfe + i + 1)
                self.trace_f.append(float(F[i]))
                self.trace_v.append(float(V[i]))
        if (self.target is not None and self.stop_reason is None and self.best is not None
                and self.cmp.reached(self.best.F, self.best.V, self.target)[0]):
            self.stop_reason = "target"

    def _better_than_best(self, F, V):
        if self.best is None:
            return np.ones(len(F), bool)
        return self.cmp.better(F, V, self.best.F, self.best.V)
