"""Native operators for GA and SA.

A native operator works on decoded values of one variable: a batch of shape (n, *shape).
A crossover takes two parent batches and returns two child batches. A mutation takes one
batch and returns one. Every operator keeps the invariants of its type (a permutation
stays a permutation, k ones stay k ones, values stay in their bounds), so
``decode(encode(y)) == y`` holds for every output.

The algorithm decodes the parents' codes, applies each variable's operator, and encodes
the result back. Partition and BinaryPartition use the permutation operators on their
ordering of items and separators. FixedSum has no native operator and keeps using the
algorithm's [0, 1] code operators on its own slice of the codes.

All random numbers come from the ``rng`` argument, and only from the methods that
``ypea.StudioRng`` offers.
"""

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np

from ..variables import (
    Binary,
    BinaryPartition,
    BinarySelection,
    Integer,
    Partition,
    Permutation,
    Real,
    Selection,
    _ranks_to_codes,
)

__all__ = [
    "uniform_crossover", "one_point_crossover",
    "order_crossover", "pmx", "cycle_crossover",
    "swap", "inversion", "insertion", "scramble",
    "RealOperators", "IntegerOperators", "PermutationOperators",
    "SelectionOperators", "BinarySelectionOperators", "PartitionOperators",
    "default_operators", "NativeOperators",
]


def _flags(rng, shape, rate):
    """Elements to change: each with probability ``rate``, and at least one per row."""
    flag = rng.random(shape) < rate
    flag[np.arange(shape[0]), rng.integers(shape[1], size=shape[0])] = True
    return flag


# array crossovers (any shape, any element type)

def uniform_crossover(x1, x2, rng):
    """Each element comes from either parent with probability 1/2."""
    swap_ = rng.random(x1.shape) < 0.5
    return np.where(swap_, x2, x1), np.where(swap_, x1, x2)


def one_point_crossover(x1, x2, rng):
    """The children swap the tails of the flattened parents after a random cut."""
    a, b = x1.reshape(len(x1), -1), x2.reshape(len(x2), -1)
    m = a.shape[1]
    if m < 2:
        return x1.copy(), x2.copy()
    tail = np.arange(m) >= rng.integers(1, m, size=len(a))[:, None]
    return np.where(tail, b, a).reshape(x1.shape), np.where(tail, a, b).reshape(x2.shape)


# permutation crossovers: one pair of rows, cut points i <= j

def order_crossover(p1, p2, i, j):
    """Order crossover (OX): keep p1[i:j], fill the rest in the order of p2 after j."""
    child = p1.copy()
    order = np.roll(p2, -j)
    pos = np.roll(np.arange(len(p1)), -j)
    pos = pos[(pos < i) | (pos >= j)]
    child[pos] = order[~np.isin(order, p1[i:j])]
    return child


def pmx(p1, p2, i, j):
    """Partially mapped crossover (PMX): keep p1[i:j], place the rest of p2 through the mapping."""
    child = p2.copy()
    child[i:j] = p1[i:j]
    where1 = np.argsort(p1)
    inseg = np.zeros(len(p1), bool)
    inseg[p1[i:j]] = True
    for k in (*range(i), *range(j, len(p1))):
        x = p2[k]
        while inseg[x]:
            x = p2[where1[x]]
        child[k] = x
    return child


def cycle_crossover(p1, p2, i=None, j=None):
    """Cycle crossover (CX): alternate cycles come from p1 and p2. The cut points are unused."""
    where1 = np.argsort(p1)
    cycle = np.full(len(p1), -1)
    c = 0
    for s in range(len(p1)):
        k = s
        while cycle[k] < 0:
            cycle[k] = c
            k = where1[p2[k]]
        c += cycle[s] == c
    return np.where(cycle % 2 == 1, p2, p1)


# permutation mutations: one row, two distinct positions i and j

def swap(x, i, j, rng):
    y = x.copy()
    y[i], y[j] = x[j], x[i]
    return y


def inversion(x, i, j, rng):
    lo, hi = min(i, j), max(i, j) + 1
    y = x.copy()
    y[lo:hi] = x[lo:hi][::-1]
    return y


def insertion(x, i, j, rng):
    """Move the item at position i to position j."""
    return np.insert(np.delete(x, i), j, x[i])


def scramble(x, i, j, rng):
    lo, hi = min(i, j), max(i, j) + 1
    y = x.copy()
    y[lo:hi] = x[lo:hi][rng.permutation(hi - lo)]
    return y


class _Operators:
    """Defaults of one variable. ``sigma`` is updated by the algorithm when its step size decays."""

    def __init__(self, var, alpha=0.1, rate=0.1, sigma=0.1):
        self.var, self.alpha, self.rate, self.sigma = var, alpha, rate, sigma

    def _rows(self, x):
        return x.reshape(-1, x.shape[-1])


class RealOperators(_Operators):
    """Blend crossover (BLX alpha) and Gaussian mutation with a step of ``sigma * (ub - lb)``."""

    def crossover(self, x1, x2, rng):
        a = rng.uniform(-self.alpha, 1 + self.alpha, x1.shape)
        lb, ub = self.var.lb, self.var.ub
        return np.clip(a * x1 + (1 - a) * x2, lb, ub), np.clip(a * x2 + (1 - a) * x1, lb, ub)

    def mutate(self, x, rng):
        v = self.var
        flat = x.reshape(len(x), -1)
        step = (_flags(rng, flat.shape, self.rate) * rng.standard_normal(flat.shape)).reshape(x.shape)
        return np.clip(x + step * self.sigma * (v.ub - v.lb), v.lb, v.ub)


class IntegerOperators(_Operators):
    """Uniform crossover. A changed element creeps (with probability ``creep``) by up to
    ``sigma * (ub - lb)`` steps, at least one, or is reset to a different value. For
    Binary the reset is a bit flip."""

    def __init__(self, var, alpha=0.1, rate=0.1, sigma=0.1, creep=0.5):
        super().__init__(var, alpha, rate, sigma)
        self.creep = 0.0 if isinstance(var, Binary) else creep

    def crossover(self, x1, x2, rng):
        return uniform_crossover(x1, x2, rng)

    def mutate(self, x, rng):
        lb, ub = self.var.lb, self.var.ub
        width = np.broadcast_to(ub - lb + 1, x.shape[1:])
        flat = x.reshape(len(x), -1)
        flag = _flags(rng, flat.shape, self.rate).reshape(x.shape)
        reset = lb + (x - lb + 1 + rng.integers(np.maximum(width - 1, 1), size=x.shape)) % width
        if self.creep:
            s = np.maximum(np.round(self.sigma * (width - 1)), 1).astype(np.int64)
            step = 1 + rng.integers(s, size=x.shape)
            step = np.where(rng.random(x.shape) < 0.5, -step, step)
            creep = np.clip(x + step, lb, ub)
            reset = np.where(rng.random(x.shape) < self.creep, creep, reset)
        return np.where(flag, reset, x)


class PermutationOperators(_Operators):
    """For each pair one crossover, and for each row one move, picked at random from the lists."""

    def __init__(self, var, alpha=0.1, rate=0.1, sigma=0.1,
                 crossovers=(order_crossover, pmx, cycle_crossover), mutations=(swap, inversion, insertion)):
        super().__init__(var, alpha, rate, sigma)
        self.crossovers, self.mutations = tuple(crossovers), tuple(mutations)

    def crossover(self, x1, x2, rng):
        a, b = self._rows(x1), self._rows(x2)
        n, m = a.shape
        which = rng.integers(len(self.crossovers), size=n)
        cuts = np.sort(rng.integers(m + 1, size=(n, 2)), axis=1)
        c1, c2 = a.copy(), b.copy()
        for r in range(n):
            op, (i, j) = self.crossovers[which[r]], cuts[r]
            c1[r], c2[r] = op(a[r], b[r], i, j), op(b[r], a[r], i, j)
        return c1.reshape(x1.shape), c2.reshape(x2.shape)

    def mutate(self, x, rng):
        a = self._rows(x)
        n, m = a.shape
        if m < 2:
            return x.copy()
        which = rng.integers(len(self.mutations), size=n)
        i = rng.integers(m, size=n)
        j = (i + 1 + rng.integers(m - 1, size=n)) % m
        y = np.array([self.mutations[which[r]](a[r], i[r], j[r], rng) for r in range(n)])
        return y.reshape(x.shape)


class PartitionOperators(PermutationOperators):
    """Partition and BinaryPartition: the permutation operators on the ordering of n items and
    k - 1 separators that the groups come from.

    The ordering is rebuilt as the variable's ``encode`` builds it: the items of group 0, then
    separator n, the items of group 1, separator n + 1, and so on. So separators are numbered
    by group position, and order crossover and PMX treat them as distinct values.
    """

    def _orders(self, values):
        codes = self.var.encode(values).reshape(-1, self.var.row_code_size)
        return np.argsort(codes, axis=1, kind="stable")

    def _values(self, orders):
        return self.var.decode(_ranks_to_codes(orders, self.var.row_code_size).reshape(-1, self.var.code_size))

    def crossover(self, x1, x2, rng):
        c1, c2 = super().crossover(self._orders(x1), self._orders(x2), rng)
        return self._values(c1), self._values(c2)

    def mutate(self, x, rng):
        return self._values(super().mutate(self._orders(x), rng))


class SelectionOperators(_Operators):
    """Ordered selection of k distinct items. Crossover takes an item from the other parent
    at a random half of the positions, when the child does not hold it yet. Mutation
    replaces one item with an unused one, or swaps two positions."""

    def crossover(self, x1, x2, rng):
        a, b = self._rows(x1), self._rows(x2)
        take = rng.random(a.shape) < 0.5
        c1, c2 = a.copy(), b.copy()
        for r in range(len(a)):
            for k in np.flatnonzero(take[r]):
                if b[r, k] not in c1[r]:
                    c1[r, k] = b[r, k]
                if a[r, k] not in c2[r]:
                    c2[r, k] = a[r, k]
        return c1.reshape(x1.shape), c2.reshape(x2.shape)

    def mutate(self, x, rng):
        a = self._rows(x)
        rows, k = a.shape
        n = self.var.n
        replace = (rng.random(rows) < 0.5) | (k == 1)
        pos = rng.integers(k, size=rows)
        other = rng.integers(max(n - k, 1), size=rows)
        pos2 = (pos + 1 + rng.integers(max(k - 1, 1), size=rows)) % k
        y = a.copy()
        for r in range(rows):
            if replace[r] and n > k:
                y[r, pos[r]] = np.setdiff1d(np.arange(n), a[r])[other[r]]
            elif k > 1:
                y[r, pos[r]], y[r, pos2[r]] = a[r, pos2[r]], a[r, pos[r]]
        return y.reshape(x.shape)


class BinarySelectionOperators(_Operators):
    """0/1 mask with exactly k ones. Crossover keeps the common ones and shares the others
    at random between the two children. Mutation swaps a one and a zero."""

    def crossover(self, x1, x2, rng):
        a, b = self._rows(x1), self._rows(x2)
        c1, c2 = a & b, a & b
        for r in range(len(a)):
            diff = np.flatnonzero(a[r] ^ b[r])[rng.permutation(int((a[r] ^ b[r]).sum()))]
            half = len(diff) // 2
            c1[r, diff[:half]] = 1
            c2[r, diff[half:]] = 1
        return c1.reshape(x1.shape), c2.reshape(x2.shape)

    def mutate(self, x, rng):
        a = self._rows(x)
        rows, n = a.shape
        k = self.var.k
        if k == n:
            return x.copy()
        one, zero = rng.integers(k, size=rows), rng.integers(n - k, size=rows)
        y = a.copy()
        for r in range(rows):
            y[r, np.flatnonzero(a[r])[one[r]]] = 0
            y[r, np.flatnonzero(1 - a[r])[zero[r]]] = 1
        return y.reshape(x.shape)


def default_operators(var, **params):
    """The native operators of a variable, or None if its type has none."""
    for cls, ops in ((Real, RealOperators), (Integer, IntegerOperators), (Permutation, PermutationOperators),
                     (Selection, SelectionOperators), (BinarySelection, BinarySelectionOperators),
                     (Partition, PartitionOperators), (BinaryPartition, PartitionOperators)):
        if type(var) is cls or (cls is Integer and type(var) is Binary):
            return ops(var, **params)
    return None


def _user(ops):
    if ops is None or not isinstance(ops, tuple):
        return ops
    return SimpleNamespace(crossover=ops[0], mutate=ops[1] if len(ops) > 1 else None)


class NativeOperators:
    """The operators of one run: the user's ones where given, otherwise the defaults,
    otherwise the algorithm's code operators on that variable's slice of the codes."""

    def __init__(self, problem, operators=None, **params):
        operators = dict(operators or {})
        unknown = set(operators) - {v.name for v in problem.vars}
        if unknown:
            raise ValueError(f"operators given for unknown variables: {sorted(unknown)}")
        self.vars = problem.vars
        self.slices = list(pairwise(np.cumsum([0, *(v.code_size for v in problem.vars)])))
        self.defaults = [default_operators(v, **params) for v in self.vars]
        self.user = [_user(operators.get(v.name)) for v in self.vars]

    def set_sigma(self, sigma):
        for d in self.defaults:
            if d is not None:
                d.sigma = sigma

    def apply(self, method, parents, rng, fallback):
        """Apply ``method`` ("crossover" or "mutate") to code batches ``parents`` and return
        code batches. ``fallback(*code_slices)`` returns a tuple of code slices."""
        out = [np.empty_like(p) for p in parents]
        for v, (a, b), user, default in zip(self.vars, self.slices, self.user, self.defaults, strict=True):
            fn = getattr(user, method, None) or getattr(default, method, None)
            parts = [p[:, a:b] for p in parents]
            if fn is None:
                ys = fallback(*parts)
            else:
                ys = fn(*(v.decode(q) for q in parts), rng)
                ys = [v.encode(y) for y in (ys if method == "crossover" else (ys,))]
            for o, y in zip(out, ys, strict=True):
                o[:, a:b] = y
        return out
