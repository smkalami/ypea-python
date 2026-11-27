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
    objective : callable or None
        Called with a decoded ``Solution``. Returns the objective value ``f``, or
        ``(f, g)`` where ``g`` is a constraint value or a list of them, each satisfied
        when ``g <= 0``, or ``(f, g, h)`` where ``h`` is an equality constraint value or a
        list of them, each satisfied when ``h`` is close to 0 (see ``equality_atol``).
        ``g`` may be None when there are only equalities. With several objectives ``f`` is a list or array of ``n_obj``
        values (not a tuple, because a tuple means ``(f, g)``). None for a problem whose
        solutions you evaluate yourself with ``Algorithm.ask_tell`` (``solve`` then
        raises an error).
    sense : "min" or "max", or a list with one of them per objective
        A single string applies to every objective.
    n_obj : int, optional
        Number of objectives (default 1, or the length of ``sense`` when it is a list).
        With more than one, the problem needs a multi-objective algorithm, or
        ``ypea.scalarize`` to turn it into a single objective problem.
    vectorized : bool
        If True, ``objective`` is called once per population with a ``Solution`` whose
        attributes have a leading population axis, and returns arrays: ``f`` with shape
        (n,) (or (n, n_obj) with several objectives), and ``g`` with shape (n,) or
        (n, number of constraints).
    constraints : "feasibility" or "penalty"
        How constrained solutions are compared (see ``ypea.comparison``).
    penalty : float
        Penalty coefficient, used with ``constraints="penalty"``.
    equality_atol, equality_rtol, equality_scale : float, float, float or list of float
        An equality ``h = 0`` counts as satisfied when ``|h| <= atol + rtol * scale``, as
        in ``numpy.isclose``. Otherwise its violation ``|h| - (atol + rtol * scale)`` adds
        to the total violation. ``equality_scale`` is the typical size of each equality
        (one number for all, or one per equality). The defaults are 1e-4 (the CEC rule),
        1e-6 and 1.
    constraint_scaling : bool or list of float
        How the violations of the constraints are added up. With True (the default), each
        constraint is divided by its largest violation among the first solutions of a run
        (at least 20, see the user guide), so constraints of different sizes count alike, as
        Deb (2000) recommends. The scales then stay fixed for the run and are kept in
        ``Result.constraint_scale``. False adds the raw violations. A list gives fixed
        scales, one per constraint (the inequalities, then the equalities).
    name : str, optional
    optimum : float, optional
        Known optimal objective value. Results then report the error ``|f - optimum|``.
    info : dict, optional
        Free metadata, for example ``{"x_opt": ..., "source": ...}`` for benchmarks and test
        problems. Values should be plain Python or NumPy data, so the problem can be pickled.
    """

    def __init__(self, vars, objective, sense="min", *, n_obj=None, vectorized=False,
                 constraints="feasibility", penalty=1e6, equality_atol=1e-4, equality_rtol=1e-6,
                 equality_scale=1.0, constraint_scaling=True, name=None, optimum=None, info=None):
        self.vars = [vars] if isinstance(vars, Variable) else list(vars)
        if not self.vars or not all(isinstance(v, Variable) for v in self.vars):
            raise TypeError("vars must be a Variable or a non empty list of Variables")
        names = [v.name for v in self.vars]
        if len(set(names)) != len(names):
            raise ValueError(f"variable names must be unique, got {names}")
        if objective is not None and not callable(objective):
            raise TypeError("objective must be callable or None")
        if vectorized and not all(v.array_valued for v in self.vars):
            raise ValueError("Partition variables decode to lists, so the problem cannot be vectorized")
        self.objective = objective
        senses = [sense] if isinstance(sense, str) else list(sense)
        if n_obj is None:
            n_obj = len(senses)
        if isinstance(n_obj, bool) or not isinstance(n_obj, (int, np.integer)) or n_obj < 1:
            raise ValueError(f"n_obj must be a positive integer, got {n_obj!r}")
        if len(senses) == 1:
            senses = senses * int(n_obj)
        if len(senses) != n_obj:
            raise ValueError(f"sense has {len(senses)} entries, but the problem has {n_obj} objectives")
        self.n_obj = int(n_obj)
        #: "min" or "max" with one objective, a list of them with several
        self.sense = parse_sense(senses[0]) if self.n_obj == 1 else [parse_sense(s) for s in senses]
        self.vectorized = bool(vectorized)
        self.constraints = constraints
        self.penalty = penalty
        self.comparator = Comparator(self.sense, constraints, penalty)
        scale = np.asarray(equality_scale, dtype=float)
        if not (np.isfinite(equality_atol) and equality_atol >= 0 and np.isfinite(equality_rtol)
                and equality_rtol >= 0 and scale.ndim <= 1 and np.all(np.isfinite(scale)) and np.all(scale > 0)):
            raise ValueError("equality_atol and equality_rtol must be non negative, "
                             "and equality_scale a positive number or a list of them")
        self.equality_atol, self.equality_rtol = float(equality_atol), float(equality_rtol)
        self.equality_scale = equality_scale
        #: (atol, rtol, scale) of the equality constraints
        self.equality_tolerance = (self.equality_atol, self.equality_rtol, scale)
        if constraint_scaling is not True and constraint_scaling is not False:
            cs = np.asarray(constraint_scaling, dtype=float)
            if cs.ndim != 1 or not cs.size or not (np.all(np.isfinite(cs)) and np.all(cs > 0)):
                raise ValueError("constraint_scaling must be True, False or a list of positive numbers, "
                                 "one per constraint (the inequalities, then the equalities)")
            constraint_scaling = cs
        self.constraint_scaling = constraint_scaling
        self.name = name or getattr(objective, "__name__", "problem")
        self.optimum = optimum
        self.info = dict(info or {})
        sizes = [v.code_size for v in self.vars]
        self._slices = np.cumsum([0, *sizes])
        self.code_size = int(self._slices[-1])

    def __repr__(self):
        n = "" if self.n_obj == 1 else f", n_obj={self.n_obj}"
        return f"Problem({self.name!r}, sense={self.sense!r}{n}, vars={self.vars})"

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
        """Objective value (an array with several objectives) and total violation of one
        decoded solution."""
        if self.objective is None:
            raise ValueError("this problem has no objective")
        return parse_result(self.objective(solution), self.n_obj, self.equality_tolerance)


def equality_violation(h, eq=None, n=None):
    return _equality_parts(h, eq, n).sum(axis=-1) if n is not None else float(_equality_parts(h, eq).sum())


def _equality_parts(h, eq=None, n=None):
    """Total violation of equality constraints ``h(x) = 0``.

    Each equality is satisfied when ``|h| <= atol + rtol * scale``, and otherwise adds
    ``|h| - (atol + rtol * scale)``. ``eq`` is ``(atol, rtol, scale)``, by default
    ``(1e-4, 1e-6, 1)``. With ``n`` (a batch), ``h`` has shape (n,) or (n, k) and the
    result has shape (n,). Otherwise ``h`` is one value or a list and the result a number.
    ``_equality_parts`` gives the violation of each equality, before the sum.
    """
    atol, rtol, scale = eq or (1e-4, 1e-6, 1.0)
    h = np.asarray(h, dtype=float)
    if n is None:
        h = h.ravel()
    else:
        if h.ndim == 0 or h.shape[0] != n:
            raise ValueError(f"vectorized equality values must have shape ({n},) or ({n}, k), got {h.shape}")
        h = h.reshape(n, -1)
    try:
        tol = atol + rtol * np.broadcast_to(scale, h.shape[-1:])
    except ValueError:
        raise ValueError(f"equality_scale has {np.size(scale)} entries, but the objective returned "
                         f"{h.shape[-1]} equality constraints") from None
    return np.maximum(np.abs(h) - tol, 0)


def _inequality(g):
    return 0.0 if g is None else float(np.maximum(np.asarray(g, dtype=float), 0).sum())


def parse_result(out, n_obj=1, eq=None, parts=False):
    """Turn an objective return value, ``f``, ``(f, g)`` or ``(f, g, h)``, into (f, violation).
    f is an array with several objectives. ``eq`` holds the equality tolerances. With
    ``parts``, the violation is an array with one entry per constraint (the inequalities,
    then the equalities), before the sum."""
    if parts:
        if isinstance(out, tuple) and len(out) == 3:
            f, g, h = out
        elif isinstance(out, tuple):
            (f, g), h = _split(out), None
        else:
            f, g, h = out, None, None
        f = _scalar(f) if n_obj == 1 else _vector(f, n_obj)
        gp = np.zeros(0) if g is None else np.maximum(np.asarray(g, dtype=float).ravel(), 0)
        return f, (gp if h is None else np.concatenate([gp, _equality_parts(h, eq)]))
    if isinstance(out, tuple) and len(out) == 3:
        f, g, h = out
        v = _inequality(g) + equality_violation(h, eq)
        return (_scalar(f) if n_obj == 1 else _vector(f, n_obj)), v
    if n_obj > 1:
        f, g = _split(out)
        return _vector(f, n_obj), _inequality(g)
    if isinstance(out, tuple):
        f, g = out
        g = np.asarray(g, dtype=float)
        return _scalar(f), float(np.maximum(g, 0).sum())
    return _scalar(out), 0.0


def _split(out):
    """(f, g) of a return value with several objectives, g None without constraints."""
    if not isinstance(out, tuple):
        return out, None
    if len(out) != 2:
        raise ValueError("a tuple return value means (f, g) or (f, g, h): "
                         "return several objective values as a list or array")
    return out


def _vector(f, n_obj):
    a = np.asarray(f, dtype=float).ravel()
    if a.size != n_obj:
        raise ValueError(f"the objective must return {n_obj} values as a list or array (a tuple means (f, g)), "
                         f"got {a.size}")
    return a


def _scalar(f):
    if type(f) is float or type(f) is np.float64:
        return float(f)
    try:
        return float(np.asarray(f, dtype=float).item())
    except ValueError:
        raise ValueError(f"the objective must return one number, got shape {np.shape(f)}") from None


def _batch_objectives(f, n, n_obj):
    f = np.asarray(f, dtype=float)
    if n_obj == 1:
        return f.reshape(n)
    if f.shape != (n, n_obj):
        raise ValueError(f"vectorized objective values must have shape ({n}, {n_obj}), got {f.shape}")
    return f


def _batch_inequality_parts(g, n):
    g = np.asarray(g, dtype=float)
    if g.ndim == 0 or g.shape[0] != n:
        raise ValueError(f"vectorized constraint values must have shape ({n},) or ({n}, m), got {g.shape}")
    return np.maximum(g.reshape(n, -1), 0)


def _batch_inequality(g, n):
    if g is None:
        return np.zeros(n)
    g = np.asarray(g, dtype=float)
    if g.ndim == 0 or g.shape[0] != n:
        raise ValueError(f"vectorized constraint values must have shape ({n},) or ({n}, m), got {g.shape}")
    return np.maximum(g.reshape(n, -1), 0).sum(axis=1)


def parse_batch(out, n, n_obj=1, eq=None, parts=False):
    """Turn a vectorized objective return value, ``F``, ``(F, G)`` or ``(F, G, H)``, into
    arrays F (n,) (or (n, n_obj)) and V (n,). With ``parts``, V has shape (n, c), one column
    per constraint (the inequalities, then the equalities), before the sum."""
    if parts:
        if isinstance(out, tuple):
            f, g, h = out if len(out) == 3 else (*_split(out), None)
        else:
            f, g, h = out, None, None
        cols = [] if g is None else [_batch_inequality_parts(g, n)]
        if h is not None:
            cols.append(_equality_parts(h, eq, n))
        return _batch_objectives(f, n, n_obj), (np.concatenate(cols, axis=1) if cols else np.zeros((n, 0)))
    if isinstance(out, tuple) and len(out) == 3:
        f, g, h = out
        return _batch_objectives(f, n, n_obj), _batch_inequality(g, n) + equality_violation(h, eq, n)
    if n_obj > 1:
        f, g = _split(out)
        return _batch_objectives(f, n, n_obj), _batch_inequality(g, n)
    if isinstance(out, tuple):
        f, g = out
        g = np.asarray(g, dtype=float)
        if g.ndim == 0 or g.shape[0] != n:
            raise ValueError(f"vectorized constraint values must have shape ({n},) or ({n}, m), got {g.shape}")
        g = g.reshape(n, -1)
        return np.asarray(f, dtype=float).reshape(n), np.maximum(g, 0).sum(axis=1)
    return np.asarray(out, dtype=float).reshape(n), np.zeros(n)
