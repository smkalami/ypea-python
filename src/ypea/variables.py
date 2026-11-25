"""Decision variable types.

Every variable is stored as a flat block of codes in [0, 1] (random keys). A variable
knows how many codes it needs, how to turn codes into values (decode), and how to turn
values back into codes (encode). Decoding is batch first: codes have shape
(n, code_size), and values have shape (n, *shape). Partition variables return lists,
because their groups have different sizes.

Indices in decoded permutations, selections and partitions are 0 based.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Variable",
    "Real",
    "Integer",
    "Binary",
    "Permutation",
    "Selection",
    "BinarySelection",
    "Partition",
    "BinaryPartition",
    "FixedSum",
]


def _shape(shape) -> tuple[int, ...]:
    if shape is None:
        return ()
    if np.isscalar(shape):
        return (int(shape),)
    return tuple(int(s) for s in shape)


class Variable:
    """Base class. Subclasses set ``name``, ``shape`` and ``code_size``."""

    #: True if ``decode`` returns an array (so the variable works in vectorized problems).
    array_valued = True

    def __init__(self, name: str):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"variable name must be a valid identifier, got {name!r}")
        self.name = name

    def decode(self, codes: np.ndarray):
        raise NotImplementedError

    def encode(self, values) -> np.ndarray:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator | None = None, n: int | None = None):
        """Random value(s) of this variable. One value if ``n`` is None."""
        rng = np.random.default_rng(rng)
        values = self.decode(rng.random((1 if n is None else n, self.code_size)))
        return values[0] if n is None else values

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, shape={self.shape})"


class Real(Variable):
    """Continuous values in [lb, ub]. Bounds may be arrays. The shape follows them if not given."""

    def __init__(self, name, shape=None, lb=0.0, ub=1.0):
        super().__init__(name)
        lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
        self.shape = _shape(shape) if shape is not None else np.broadcast_shapes(lb.shape, ub.shape)
        self.lb = np.broadcast_to(lb, self.shape).copy()
        self.ub = np.broadcast_to(ub, self.shape).copy()
        if np.any(self.ub < self.lb):
            raise ValueError(f"{name}: upper bound is below lower bound")
        self.code_size = int(np.prod(self.shape))

    def decode(self, codes):
        c = codes.reshape(len(codes), *self.shape)
        return self.lb + (self.ub - self.lb) * c

    def encode(self, values):
        v = np.asarray(values, dtype=float).reshape(-1, *self.shape)
        span = np.where(self.ub > self.lb, self.ub - self.lb, 1.0)
        return np.clip((v - self.lb) / span, 0, 1).reshape(len(v), -1)


class Integer(Variable):
    """Integer values in [lb, ub], both included."""

    def __init__(self, name, shape=None, lb=0, ub=1):
        super().__init__(name)
        lb, ub = np.ceil(np.asarray(lb)), np.floor(np.asarray(ub))
        self.shape = _shape(shape) if shape is not None else np.broadcast_shapes(lb.shape, ub.shape)
        self.lb = np.broadcast_to(lb, self.shape).astype(np.int64)
        self.ub = np.broadcast_to(ub, self.shape).astype(np.int64)
        if np.any(self.ub < self.lb):
            raise ValueError(f"{name}: upper bound is below lower bound")
        self.code_size = int(np.prod(self.shape))

    def decode(self, codes):
        c = codes.reshape(len(codes), *self.shape)
        width = self.ub - self.lb + 1
        # min() keeps code 1.0 at ub (MATLAB decoded it to ub + 1)
        return self.lb + np.minimum(np.floor(width * c).astype(np.int64), width - 1)

    def encode(self, values):
        v = np.asarray(values).reshape(-1, *self.shape)
        return ((v - self.lb + 0.5) / (self.ub - self.lb + 1)).reshape(len(v), -1)


class Binary(Integer):
    """Values 0 or 1."""

    def __init__(self, name, shape=None):
        super().__init__(name, shape=() if shape is None else shape, lb=0, ub=1)


class _RowVariable(Variable):
    """A variable made of ``count`` independent rows (for example several permutations).

    Subclasses implement ``_decode_rows`` and ``_encode_rows`` on 2D arrays of rows.
    """

    def __init__(self, name, row_code_size, count):
        super().__init__(name)
        self.count = None if count is None else int(count)
        self.rows = 1 if count is None else self.count
        self.row_code_size = int(row_code_size)
        self.code_size = self.rows * self.row_code_size

    def decode(self, codes):
        n = len(codes)
        out = self._decode_rows(codes.reshape(n * self.rows, self.row_code_size))
        if isinstance(out, list):
            if self.count is None:
                return out
            return [out[i * self.rows:(i + 1) * self.rows] for i in range(n)]
        return out.reshape(n, *self.shape)

    def encode(self, values):
        if self.array_valued:
            v = np.asarray(values).reshape(-1, *self.row_shape)
        elif self.count is None:
            v = list(values)
        else:
            v = [row for item in values for row in item]
        rows = self._encode_rows(v)
        return rows.reshape(-1, self.code_size)

    @property
    def shape(self):
        return self.row_shape if self.count is None else (self.count, *self.row_shape)


def _ranks_to_codes(order: np.ndarray, size: int) -> np.ndarray:
    """Codes whose stable argsort starts with ``order`` (an integer array per row)."""
    k = order.shape[1]
    codes = np.empty((len(order), size))
    rows = np.arange(len(order))[:, None]
    codes[rows, order] = (np.arange(k) + 0.5) / size
    if k < size:
        # remaining items get larger codes, in index order
        mask = np.ones((len(order), size), bool)
        mask[rows, order] = False
        rest = (k + np.arange(size - k) + 0.5) / size
        codes[mask] = np.tile(rest, len(order))
    return codes


class Permutation(_RowVariable):
    """A permutation of 0..n-1. With ``count``, a (count, n) array with one permutation per row."""

    def __init__(self, name, n, count=None):
        super().__init__(name, n, count)
        self.n = int(n)
        self.row_shape = (self.n,)

    def _decode_rows(self, codes):
        return np.argsort(codes, axis=1, kind="stable")

    def _encode_rows(self, perms):
        return _ranks_to_codes(perms, self.n)


class Selection(_RowVariable):
    """An ordered selection of k distinct items out of n (0 based indices)."""

    def __init__(self, name, n, k, count=None):
        super().__init__(name, n, count)
        self.n, self.k = int(n), int(k)
        if not 1 <= self.k <= self.n:
            raise ValueError(f"{name}: k must be between 1 and n")
        self.row_shape = (self.k,)

    def _decode_rows(self, codes):
        return np.argsort(codes, axis=1, kind="stable")[:, : self.k]

    def _encode_rows(self, sel):
        return _ranks_to_codes(sel, self.n)


class BinarySelection(_RowVariable):
    """A 0/1 mask of length n with exactly k ones."""

    def __init__(self, name, n, k, count=None):
        super().__init__(name, n, count)
        self.n, self.k = int(n), int(k)
        if not 1 <= self.k <= self.n:
            raise ValueError(f"{name}: k must be between 1 and n")
        self.row_shape = (self.n,)

    def _decode_rows(self, codes):
        idx = np.argsort(codes, axis=1, kind="stable")[:, : self.k]
        mask = np.zeros(codes.shape, dtype=np.int64)
        np.put_along_axis(mask, idx, 1, axis=1)
        return mask

    def _encode_rows(self, masks):
        order = np.argsort(-masks, axis=1, kind="stable")[:, : self.k]
        return _ranks_to_codes(order, self.n)


class Partition(_RowVariable):
    """Split items 0..n-1 into k groups (some may be empty). Decodes to a list of k index arrays.

    The code has n + k - 1 keys: n items and k - 1 separators, sorted together.
    """

    array_valued = False

    def __init__(self, name, n, k, count=None):
        super().__init__(name, n + k - 1, count)
        self.n, self.k = int(n), int(k)
        if self.k < 1:
            raise ValueError(f"{name}: k must be at least 1")
        self.row_shape = (self.k,)

    def _decode_rows(self, codes):
        order = np.argsort(codes, axis=1, kind="stable")
        return [_groups(row, np.flatnonzero(row >= self.n)) for row in order]

    def _encode_rows(self, partitions):
        orders = []
        for groups in partitions:
            seq = []
            for j, g in enumerate(groups):
                if j:
                    seq.append(self.n + j - 1)
                seq.extend(int(i) for i in g)
            orders.append(seq)
        return _ranks_to_codes(np.array(orders, dtype=np.int64), self.row_code_size)


def _groups(row, cuts):
    groups, start = [], 0
    for c in cuts:
        groups.append(row[start:c])
        start = c + 1
    groups.append(row[start:])
    return groups


class BinaryPartition(Partition):
    """Like Partition, but decodes to a (k, n) 0/1 assignment matrix."""

    array_valued = True

    def __init__(self, name, n, k, count=None):
        super().__init__(name, n, k, count)
        self.row_shape = (self.k, self.n)

    def _decode_rows(self, codes):
        out = np.zeros((len(codes), self.k, self.n), dtype=np.int64)
        for r, groups in enumerate(super()._decode_rows(codes)):
            for j, g in enumerate(groups):
                out[r, j, g] = 1
        return out

    def _encode_rows(self, mats):
        mats = np.asarray(mats)
        return super()._encode_rows([[np.flatnonzero(m[j]) for j in range(self.k)] for m in mats])


class FixedSum(Variable):
    """Non negative reals with a fixed sum.

    ``axis=None`` fixes the sum of all elements, ``axis=1`` the sum of each row, and
    ``axis=0`` the sum of each column. ``target`` is a scalar, or one value per row or column.
    """

    def __init__(self, name, shape, target=1.0, axis=None):
        super().__init__(name)
        self.shape = _shape(shape)
        if axis not in (None, 0, 1):
            raise ValueError(f"{name}: axis must be None, 0 or 1")
        self.axis = axis
        self.target = np.asarray(target, dtype=float)
        if axis is None and self.target.ndim:
            raise ValueError(f"{name}: target must be a scalar when axis is None")
        if axis is not None:
            if len(self.shape) != 2:
                raise ValueError(f"{name}: axis needs a 2D shape")
            m = self.shape[1 - axis]
            self.target = np.broadcast_to(self.target, (m,)).copy()
        self.code_size = int(np.prod(self.shape))

    def _sum_axes(self):
        return tuple(range(1, len(self.shape) + 1)) if self.axis is None else self.axis + 1

    def _target(self):
        if self.axis is None:
            return self.target
        return np.expand_dims(self.target, self.axis)

    def decode(self, codes):
        c = codes.reshape(len(codes), *self.shape)
        s = c.sum(axis=self._sum_axes(), keepdims=True)
        # all zero codes would give 0/0: share the target equally instead
        c = np.where(s > 0, c, 1.0)
        s = np.where(s > 0, s, c.sum(axis=self._sum_axes(), keepdims=True))
        return self._target() * c / s

    def encode(self, values):
        v = np.abs(np.asarray(values, dtype=float).reshape(-1, *self.shape))
        m = v.max(axis=self._sum_axes(), keepdims=True)
        return (v / np.where(m > 0, m, 1.0)).reshape(len(v), -1)
