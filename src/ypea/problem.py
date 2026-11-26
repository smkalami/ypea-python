"""Optimization problem: decision variables, objective, and how solutions are compared."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .comparison import Comparator, parse_sense
from .variables import Variable


class Solution(SimpleNamespace):
    """A decoded solution. Each variable is an attribute (``sol.x``) and also a key (``sol["x"]``)."""

    def __getitem__(self, key):
        return getattr(self, key)


class Problem:
    """An optimization problem.

    Parameters
    ----------
    vars : Variable or list of Variable
        The decision variables. Names must be unique.
    objective : callable
        Called with a decoded ``Solution``. Returns the objective value ``f``, or
        ``(f, g)`` where ``g`` is a constraint value or a list of them, each satisfied
        when ``g <= 0``.
    sense : "min" or "max"
    vectorized : bool
        If True, ``objective`` is called once per population with a ``Solution`` whose
        attributes have a leading population axis, and returns arrays: ``f`` with shape
        (n,), and ``g`` with shape (n,) or (n, number of constraints).
    constraints : "feasibility" or "penalty"
        How constrained solutions are compared (see ``ypea.comparison``).
    penalty : float
        Penalty coefficient, used with ``constraints="penalty"``.
    name : str, optional
    optimum : float, optional
        Known optimal objective value. Results then report the error ``|f - optimum|``.
    info : dict, optional
        Free metadata, for example ``{"x_opt": ..., "source": ...}`` for benchmarks and test
        problems. Values should be plain Python or NumPy data, so the problem can be pickled.
    """

    def __init__(self, vars, objective, sense="min", *, vectorized=False,
                 constraints="feasibility", penalty=1e6, name=None, optimum=None, info=None):
        self.vars = [vars] if isinstance(vars, Variable) else list(vars)
        if not self.vars or not all(isinstance(v, Variable) for v in self.vars):
            raise TypeError("vars must be a Variable or a non empty list of Variables")
        names = [v.name for v in self.vars]
        if len(set(names)) != len(names):
            raise ValueError(f"variable names must be unique, got {names}")
        if not callable(objective):
            raise TypeError("objective must be callable")
        if vectorized and not all(v.array_valued for v in self.vars):
            raise ValueError("Partition variables decode to lists, so the problem cannot be vectorized")
        self.objective = objective
        self.sense = parse_sense(sense)
        self.vectorized = bool(vectorized)
        self.constraints = constraints
        self.penalty = penalty
        self.comparator = Comparator(self.sense, constraints, penalty)
        self.name = name or getattr(objective, "__name__", "problem")
        self.optimum = optimum
        self.info = dict(info or {})
        sizes = [v.code_size for v in self.vars]
        self._slices = np.cumsum([0, *sizes])
        self.code_size = int(self._slices[-1])

    def __repr__(self):
        return f"Problem({self.name!r}, sense={self.sense!r}, vars={self.vars})"

    # decoding and encoding

    def decode(self, X):
        """Decode codes of shape (n, code_size). Returns a batch ``Solution`` if vectorized,
        otherwise a list of ``Solution``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.code_size:
            raise ValueError(f"expected codes of length {self.code_size}, got {X.shape[1]}")
        parts = {v.name: v.decode(X[:, a:b]) for v, a, b in
                 zip(self.vars, self._slices[:-1], self._slices[1:], strict=True)}
        if self.vectorized:
            return Solution(**parts)
        # iterating a decoded array gives the same rows as indexing it
        if len(parts) == 1:
            (name, values), = parts.items()
            return [Solution(**{name: v}) for v in values]
        names = list(parts)
        return [Solution(**dict(zip(names, row, strict=True))) for row in zip(*parts.values(), strict=True)]

    def decode_one(self, x) -> Solution:
        X = np.asarray(x, dtype=float).reshape(1, -1)
        parts = {v.name: v.decode(X[:, a:b])[0] for v, a, b in
                 zip(self.vars, self._slices[:-1], self._slices[1:], strict=True)}
        return Solution(**parts)

    def encode(self, solution) -> np.ndarray:
        """Code vector of a solution given as a ``Solution`` or a dict of variable values."""
        get = solution.__getitem__ if isinstance(solution, dict) else lambda k: getattr(solution, k)
        return np.concatenate([v.encode([get(v.name)])[0] for v in self.vars])

    def random_codes(self, n, rng):
        return rng.random((n, self.code_size))

    # evaluation

    def evaluate(self, solution):
        """Objective value and total violation of one decoded solution."""
        return parse_result(self.objective(solution))


def parse_result(out):
    """Turn an objective return value into (f, violation)."""
    if isinstance(out, tuple):
        f, g = out
        g = np.asarray(g, dtype=float)
        return _scalar(f), float(np.maximum(g, 0).sum())
    return _scalar(out), 0.0


def _scalar(f):
    if type(f) is float or type(f) is np.float64:
        return float(f)
    try:
        return float(np.asarray(f, dtype=float).item())
    except ValueError:
        raise ValueError(f"the objective must return one number, got shape {np.shape(f)}") from None


def parse_batch(out, n):
    """Turn a vectorized objective return value into arrays F (n,) and V (n,)."""
    if isinstance(out, tuple):
        f, g = out
        g = np.asarray(g, dtype=float)
        if g.ndim == 0 or g.shape[0] != n:
            raise ValueError(f"vectorized constraint values must have shape ({n},) or ({n}, m), got {g.shape}")
        g = g.reshape(n, -1)
        return np.asarray(f, dtype=float).reshape(n), np.maximum(g, 0).sum(axis=1)
    return np.asarray(out, dtype=float).reshape(n), np.zeros(n)
