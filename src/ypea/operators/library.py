"""Operators that an algorithm takes as ``crossover=`` and ``mutation=`` settings.

Every operator declares where it applies with its ``space`` attribute:

* ``"code"``: it works on whole code batches of shape (n, d) in [0, 1], the universal
  search space. It works for every variable type, because each type decodes its codes.
  A code crossover is called as ``op(x1, x2, rng)`` and returns two child batches. A
  code mutation is called as ``op(x, rng)`` and returns one batch.
* ``"native"``: it works on the decoded values of one variable at a time, for the
  variable types listed in ``types``. It is called like a code operator, but with the
  decoded batch of one variable (shape (n, *shape)). Variables of other types keep the
  algorithm's own operators.

An algorithm checks the operators before a run and raises a clear error when one does
not fit (a native operator in the code space, or one whose types match no variable).

All random numbers come from the ``rng`` argument, and only from the methods that
``ypea.StudioRng`` offers. The draws have fixed shapes that do not depend on the data.
"""

from __future__ import annotations

import numpy as np


def _show(v):
    # functions and classes by name, so the text (and the key of a stored run) is stable across processes
    if callable(v) and hasattr(v, "__qualname__"):
        return f"{getattr(v, '__module__', '')}.{v.__qualname__}"
    if isinstance(v, tuple):
        return "(" + ", ".join(_show(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return repr(v)


class Operator:
    """Base class of all operators. ``space`` is "code" or "native", and ``types`` lists
    the variable types of a native operator."""

    space = "code"
    types = ()

    def __repr__(self):
        args = ", ".join(f"{k}={_show(v)}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class Crossover(Operator):
    """Base class of crossover operators, called as ``op(x1, x2, rng)``."""

    def __call__(self, x1, x2, rng):
        raise NotImplementedError


class Mutation(Operator):
    """Base class of mutation operators, called as ``op(x, rng)``."""

    def __call__(self, x, rng):
        raise NotImplementedError


class NativeCrossover(Crossover):
    """A crossover on the decoded values of the variables whose type is in ``types``.

    ``fn(x1, x2, rng)`` takes two decoded batches of one variable and returns two, for
    example ``NativeCrossover(ypea.operators.uniform_crossover, types=(ypea.Integer,))``.
    """

    space = "native"

    def __init__(self, fn, types):
        self.fn = fn
        self.types = tuple(types) if isinstance(types, (tuple, list)) else (types,)

    def __call__(self, x1, x2, rng):
        return self.fn(x1, x2, rng)


class NativeMutation(Mutation):
    """A mutation on the decoded values of the variables whose type is in ``types``.
    ``fn(x, rng)`` takes a decoded batch of one variable and returns one."""

    space = "native"

    def __init__(self, fn, types):
        self.fn = fn
        self.types = tuple(types) if isinstance(types, (tuple, list)) else (types,)

    def __call__(self, x, rng):
        return self.fn(x, rng)


class SBX(Crossover):
    """Simulated binary crossover (SBX) in the code space, bounded to [0, 1].

    Each pair of parents is crossed with probability ``prob``, and then each component
    with probability ``prob_var``. A crossed component spreads the two parent values by a
    factor drawn from a polynomial distribution with index ``eta`` (larger values keep the
    children closer to the parents). The bounded form takes the distance to the bounds
    into account, so no child leaves [0, 1], and the two children are swapped with
    probability 1/2. Components whose parent values differ by less than 1e-14 are copied.

    The defaults (``eta=20``, ``prob=0.9``) are the NSGA-II settings of Deb et al. (2002).
    The bounded form and ``prob_var=0.5`` follow Deb's NSGA-II reference code.

    References: Deb and Agrawal (1995), Simulated binary crossover for continuous search
    space, Complex Systems 9, 115 to 148. Deb, Pratap, Agarwal and Meyarivan (2002), IEEE
    Transactions on Evolutionary Computation 6(2), 182 to 197.
    """

    def __init__(self, eta=20.0, prob=0.9, prob_var=0.5):
        if not (np.isfinite(eta) and eta >= 0):
            raise ValueError("eta must be a non negative number")
        if not (0 <= prob <= 1 and 0 <= prob_var <= 1):
            raise ValueError("prob and prob_var must be in [0, 1]")
        self.eta, self.prob, self.prob_var = float(eta), float(prob), float(prob_var)

    def __call__(self, x1, x2, rng):
        x1, x2 = np.clip(x1, 0.0, 1.0), np.clip(x2, 0.0, 1.0)
        n, d = x1.shape
        pair = rng.random(n) <= self.prob
        var = rng.random((n, d)) <= self.prob_var
        u = rng.random((n, d))
        swap = rng.random((n, d)) <= 0.5
        lo, hi = np.minimum(x1, x2), np.maximum(x1, x2)
        diff = hi - lo
        cross = pair[:, None] & var & (diff > 1e-14)
        diff = np.where(cross, diff, 1.0)  # avoids dividing by zero where nothing is crossed
        e = self.eta + 1

        def spread(beta):
            alpha = 2 - beta ** -e
            return np.where(u <= 1 / alpha, (u * alpha) ** (1 / e), (1 / (2 - u * alpha)) ** (1 / e))

        c1 = np.clip(0.5 * (lo + hi - spread(1 + 2 * lo / diff) * diff), 0.0, 1.0)
        c2 = np.clip(0.5 * (lo + hi + spread(1 + 2 * (1 - hi) / diff) * diff), 0.0, 1.0)
        y1 = np.where(cross, np.where(swap, c2, c1), x1)
        y2 = np.where(cross, np.where(swap, c1, c2), x2)
        return y1, y2


class PolynomialMutation(Mutation):
    """Polynomial mutation in the code space, bounded to [0, 1].

    Each component changes with probability ``prob`` (default 1/d for d code components).
    The change follows a polynomial distribution with index ``eta`` (larger values give
    smaller changes), scaled so that the result stays in [0, 1].

    The defaults (``eta=20``, ``prob=1/d``) are the NSGA-II settings of Deb et al. (2002).
    The bounded form follows Deb's NSGA-II reference code.

    References: Deb and Goyal (1996), A combined genetic adaptive search (GeneAS) for
    engineering design, Computer Science and Informatics 26(4), 30 to 45. Deb et al. (2002).
    """

    def __init__(self, eta=20.0, prob=None):
        if not (np.isfinite(eta) and eta >= 0):
            raise ValueError("eta must be a non negative number")
        if prob is not None and not 0 <= prob <= 1:
            raise ValueError("prob must be in [0, 1] or None")
        self.eta, self.prob = float(eta), prob

    def __call__(self, x, rng):
        x = np.clip(x, 0.0, 1.0)
        n, d = x.shape
        p = 1 / d if self.prob is None else self.prob
        flag = rng.random((n, d)) <= p
        u = rng.random((n, d))
        e = self.eta + 1
        low = u <= 0.5
        val = np.where(low, 2 * u + (1 - 2 * u) * (1 - x) ** e, 2 * (1 - u) + 2 * (u - 0.5) * x ** e)
        delta = np.where(low, val ** (1 / e) - 1, 1 - val ** (1 / e))
        return np.where(flag, np.clip(x + delta, 0.0, 1.0), x)


def check_operator(op, kind, space, problem=None):
    """Check that ``op`` is a ``kind`` ("crossover" or "mutation") operator that fits the
    algorithm's ``space`` ("encoded" or "native") and, when given, the problem. None passes."""
    if op is None:
        return
    base = Mutation if kind == "mutation" else Crossover
    if not isinstance(op, base):
        raise TypeError(f"{kind} must be a ypea.operators.{base.__name__} (for example "
                        f"{'PolynomialMutation()' if kind == 'mutation' else 'SBX()'}) or None, got {op!r}")
    if op.space == "native" and space != "native":
        raise ValueError(f"{op!r} works on native values, so it needs space='native'")
    if op.space == "code" and space == "native":
        raise ValueError(f"{op!r} works on the [0, 1] codes, so it needs space='encoded' (the default)")
    if op.space == "native" and problem is not None and not any(type(v) in op.types for v in problem.vars):
        names = ", ".join(t.__name__ for t in op.types)
        raise ValueError(f"{op!r} applies to {names} variables, and this problem has none")
