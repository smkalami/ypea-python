"""Multi-objective benchmark functions: ZDT, DTLZ and WFG.

Each function takes one point (shape (d,)) or a batch (shape (..., d)) and returns the
objective values with shape (..., m). All objectives are minimized. The scalable suites
take the number of objectives ``m``. DTLZ infers k = d - m + 1 from the dimension. WFG
takes the number of position parameters ``k`` (default 2 (m - 1)) and uses the rest of
the variables as distance parameters.

``REGISTRY`` holds the bounds, the default dimension and number of objectives, and the
shape of the Pareto front of each problem. ``pareto_front(name, n_points, m)`` samples the
true front. ``das_dennis(n_points, m)`` gives uniform reference points on the simplex.

Sources:

- Zitzler, Deb and Thiele (2000), "Comparison of multiobjective evolutionary algorithms:
  Empirical results", Evolutionary Computation 8(2). ZDT1 to ZDT6.
- Deb, Thiele, Laumanns and Zitzler (2001), "Scalable test problems for evolutionary
  multi-objective optimization", TIK Technical Report 112, ETH Zurich. DTLZ1 to DTLZ7. Two
  typing errors of the report are read as in later implementations: the DTLZ1 optimum is
  x_M = 0.5 (as its g function gives), and the DTLZ5 angles theta_i are used directly, not
  multiplied by pi / 2 again.
- Huband, Hingston, Barone and While (2006), "A review of multiobjective test problems and
  a scalable test problem toolkit", IEEE Transactions on Evolutionary Computation 10(5).
  WFG1 to WFG9. The values agree with the pymoo and PlatEMO implementations, except for
  the rounding step noted in ``wfg1``.
- Das and Dennis (1998), "Normal-boundary intersection", SIAM Journal on Optimization 8(3).
  Reference points on the simplex.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

PI = np.pi


def _x(x):
    return np.asarray(x, dtype=float)


def _shape(p, a, b):
    """Objectives h_1..h_M of the product shapes from M - 1 parameters p (shape (..., M - 1)).

    h_1 = prod a(p_i), h_m = prod_{i <= M - m} a(p_i) b(p_{M - m + 1}), h_M = b(p_1).
    DTLZ2 uses a = cos and b = sin of the angles. WFG uses linear, convex and concave
    versions.
    """
    one = np.ones(p.shape[:-1] + (1,))
    head = np.cumprod(np.concatenate([one, a(p)], axis=-1), axis=-1)[..., ::-1]
    tail = np.concatenate([one, b(p)[..., ::-1]], axis=-1)
    return head * tail


def _cos_sin(theta):
    return _shape(theta, np.cos, np.sin)


# ZDT (two objectives)

def _zdt_g(x):
    return 1 + 9 * np.mean(x[..., 1:], axis=-1)


def zdt1(x):
    """ZDT1: convex front f2 = 1 - sqrt(f1). x in [0, 1]^d, d = 30."""
    x = _x(x)
    f1, g = x[..., 0], _zdt_g(x)
    return np.stack([f1, g * (1 - np.sqrt(f1 / g))], axis=-1)


def zdt2(x):
    """ZDT2: concave front f2 = 1 - f1^2. x in [0, 1]^d, d = 30."""
    x = _x(x)
    f1, g = x[..., 0], _zdt_g(x)
    return np.stack([f1, g * (1 - (f1 / g) ** 2)], axis=-1)


def zdt3(x):
    """ZDT3: front of five disconnected parts, f2 = 1 - sqrt(f1) - f1 sin(10 pi f1). d = 30."""
    x = _x(x)
    f1, g = x[..., 0], _zdt_g(x)
    return np.stack([f1, g * (1 - np.sqrt(f1 / g) - f1 / g * np.sin(10 * PI * f1))], axis=-1)


def zdt4(x):
    """ZDT4: Rastrigin type g with 21^9 local fronts. x1 in [0, 1], the rest in [-5, 5], d = 10."""
    x = _x(x)
    f1, y = x[..., 0], x[..., 1:]
    g = 1 + 10 * y.shape[-1] + np.sum(y**2 - 10 * np.cos(4 * PI * y), axis=-1)
    return np.stack([f1, g * (1 - np.sqrt(f1 / g))], axis=-1)


def _zdt5_v(u):
    return np.where(u < 5, 2 + u, 1.0)


def zdt5(x):
    """ZDT5: deceptive binary problem. x holds 0 and 1 bits, d = 30 + 5 j (default 80).

    u_1 counts the ones in the first 30 bits and u_i the ones in each next group of 5.
    f1 = 1 + u_1, g = sum v(u_i) with v(u) = 2 + u for u < 5 and v(5) = 1, f2 = g / f1.
    """
    x = _x(x)
    d = x.shape[-1]
    if d < 35 or (d - 30) % 5:
        raise ValueError("zdt5 needs d = 30 + 5 j bits with j >= 1")
    u1 = np.sum(x[..., :30], axis=-1)
    u = np.sum(x[..., 30:].reshape(x.shape[:-1] + (-1, 5)), axis=-1)
    f1 = 1 + u1
    return np.stack([f1, np.sum(_zdt5_v(u), axis=-1) / f1], axis=-1)


def zdt6(x):
    """ZDT6: nonuniform concave front f2 = 1 - f1^2 with f1 >= 0.2808. x in [0, 1]^d, d = 10."""
    x = _x(x)
    f1 = 1 - np.exp(-4 * x[..., 0]) * np.sin(6 * PI * x[..., 0]) ** 6
    g = 1 + 9 * np.mean(x[..., 1:], axis=-1) ** 0.25
    return np.stack([f1, g * (1 - (f1 / g) ** 2)], axis=-1)


# DTLZ (scalable, x in [0, 1]^d with d = m - 1 + k)

def _dtlz_split(x, m):
    x = _x(x)
    if m < 2 or x.shape[-1] < m:
        raise ValueError(f"DTLZ needs m >= 2 and d >= m, got m={m}, d={x.shape[-1]}")
    return x[..., : m - 1], x[..., m - 1:]


def _g_rastrigin(xm):
    k = xm.shape[-1]
    return 100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * PI * (xm - 0.5)), axis=-1))


def _g_sphere(xm):
    return np.sum((xm - 0.5) ** 2, axis=-1)


def dtlz1(x, m=3):
    """DTLZ1: linear front sum f = 0.5 with a multimodal g. Default k = 5."""
    xp, xm = _dtlz_split(x, m)
    g = _g_rastrigin(xm)
    return 0.5 * (1 + g)[..., None] * _shape(xp, lambda p: p, lambda p: 1 - p)


def dtlz2(x, m=3):
    """DTLZ2: spherical front sum f^2 = 1. Default k = 10."""
    xp, xm = _dtlz_split(x, m)
    return (1 + _g_sphere(xm))[..., None] * _cos_sin(xp * PI / 2)


def dtlz3(x, m=3):
    """DTLZ3: spherical front with the multimodal g of DTLZ1. Default k = 10."""
    xp, xm = _dtlz_split(x, m)
    return (1 + _g_rastrigin(xm))[..., None] * _cos_sin(xp * PI / 2)


def dtlz4(x, m=3, alpha=100.0):
    """DTLZ4: DTLZ2 with the biased mapping x_i^alpha (alpha = 100). Default k = 10."""
    xp, xm = _dtlz_split(x, m)
    return (1 + _g_sphere(xm))[..., None] * _cos_sin(xp**alpha * PI / 2)


def _dtlz5_angles(xp, g):
    theta = PI / (4 * (1 + g[..., None])) * (1 + 2 * g[..., None] * xp)
    theta[..., 0] = xp[..., 0] * PI / 2
    return theta


def dtlz5(x, m=3):
    """DTLZ5: degenerate front, a curve on the unit sphere. Default k = 10."""
    xp, xm = _dtlz_split(x, m)
    g = _g_sphere(xm)
    return (1 + g)[..., None] * _cos_sin(_dtlz5_angles(xp, g))


def dtlz6(x, m=3):
    """DTLZ6: DTLZ5 with g = sum x_i^0.1, harder to converge. Default k = 10."""
    xp, xm = _dtlz_split(x, m)
    g = np.sum(xm**0.1, axis=-1)
    return (1 + g)[..., None] * _cos_sin(_dtlz5_angles(xp, g))


def _dtlz7_t(f):
    return f / 2 * (1 + np.sin(3 * PI * f))


def dtlz7(x, m=3):
    """DTLZ7: 2^(m - 1) disconnected front regions. Default k = 20.

    f_i = x_i for i < m, g = 1 + 9 / k sum x_M, f_m = (1 + g) (m - sum f_i / (1 + g) (1 + sin(3 pi f_i))).
    """
    xp, xm = _dtlz_split(x, m)
    g = 1 + 9 * np.mean(xm, axis=-1)
    h = m - np.sum(xp / (1 + g[..., None]) * (1 + np.sin(3 * PI * xp)), axis=-1)
    return np.concatenate([xp, ((1 + g) * h)[..., None]], axis=-1)


# WFG (scalable, x_i in [0, 2 i], d = k + l)

def _s_linear(y, a=0.35):
    t = np.abs(y - a) / np.abs(np.floor(a - y) + a)
    # rounding leaves about 1e-16 at the optimum y = a (x_i = 0.7 i), and the bias y^0.02 of
    # WFG1 turns that into 0.48, so values below 1e-12 count as 0
    return np.where(t < 1e-12, 0.0, t)


def _s_decept(y, a=0.35, b=0.001, c=0.05):
    t1 = np.floor(y - a + b) * (1 - c + (a - b) / b) / (a - b)
    t2 = np.floor(a + b - y) * (1 - c + (1 - a - b) / b) / (1 - a - b)
    return 1 + (np.abs(y - a) - b) * (t1 + t2 + 1 / b)


def _s_multi(y, a, b, c):
    t = np.abs(y - c) / (2 * (np.floor(c - y) + c))
    return (1 + np.cos((4 * a + 2) * PI * (0.5 - t)) + 4 * b * t**2) / (b + 2)


def _b_flat(y, a=0.8, b=0.75, c=0.85):
    out = (a + np.minimum(0, np.floor(y - b)) * a * (b - y) / b
           - np.minimum(0, np.floor(c - y)) * (1 - a) * (y - c) / (1 - c))
    return np.clip(out, 0, 1)  # rounding can leave -2e-16 at y = 0, and the next step takes a power


def _b_param(y, u, a=0.98 / 49.98, b=0.02, c=50.0):
    return y ** (b + (c - b) * (a - (1 - 2 * u) * np.abs(np.floor(0.5 - u) + a)))


def _mean_after(y):
    """Mean of y_{i+1}..y_n for i = 1..n-1 (the last variable has none)."""
    rest = np.cumsum(y[..., ::-1], axis=-1)[..., ::-1][..., 1:]
    return rest / np.arange(y.shape[-1] - 1, 0, -1)


def _mean_before(y, start):
    """Mean of y_1..y_{i-1} for i = start + 1..n (0 based start >= 1)."""
    before = np.cumsum(y, axis=-1)[..., start - 1:-1]
    return before / np.arange(start, y.shape[-1])


def _r_sum(y, w=None):
    if w is None:
        return np.mean(y, axis=-1)
    return np.sum(y * w, axis=-1) / np.sum(w)


def _r_nonsep(y, a):
    n = y.shape[-1]
    total = np.sum(y, axis=-1)
    for k in range(1, a):
        total = total + np.sum(np.abs(y - np.roll(y, -k, axis=-1)), axis=-1)
    ca = np.ceil(a / 2)
    return total / (n / a * ca * (1 + 2 * a - 2 * ca))


def _wfg_split(x, m, k, even=False):
    x = _x(x)
    d = x.shape[-1]
    k = 2 * (m - 1) if k is None else int(k)
    if m < 2 or k < 1 or k % (m - 1) or d <= k:
        raise ValueError(f"WFG needs m >= 2, k a multiple of m - 1 and d > k, got m={m}, k={k}, d={d}")
    if even and (d - k) % 2:
        raise ValueError("WFG2 and WFG3 need an even number of distance parameters d - k")
    return x / (2 * np.arange(1, d + 1)), k


def _reduce(t, m, k, pos, dist):
    """Reduce the k position parameters to m - 1 values and the rest to one distance value."""
    g = k // (m - 1)
    tp = t[..., :k].reshape(t.shape[:-1] + (m - 1, g))
    return np.concatenate([pos(tp), dist(t[..., k:])[..., None]], axis=-1)


def _wfg_f(t, a_degenerate, shape):
    """Objectives from the reduced values t (shape (..., m)), with D = 1 and S_m = 2 m."""
    m = t.shape[-1]
    xm = t[..., -1:]
    a = np.ones(m - 1)
    if a_degenerate:
        a[1:] = 0
    p = np.maximum(xm, a) * (t[..., :-1] - 0.5) + 0.5
    return xm + 2 * np.arange(1, m + 1) * shape(p)


def _linear(p):
    return _shape(p, lambda q: q, lambda q: 1 - q)


def _convex(p):
    return _shape(p, lambda q: 1 - np.cos(q * PI / 2), lambda q: 1 - np.sin(q * PI / 2))


def _concave(p):
    return _shape(p, lambda q: np.sin(q * PI / 2), lambda q: np.cos(q * PI / 2))


def _mixed(p1):
    return 1 - p1 - np.cos(10 * PI * p1 + PI / 2) / (10 * PI)


def _disc(p1):
    return 1 - p1 * np.cos(5 * PI * p1) ** 2


def _convex_mixed(p):
    h = _convex(p)
    h[..., -1] = _mixed(p[..., 0])
    return h


def _convex_disc(p):
    h = _convex(p)
    h[..., -1] = _disc(p[..., 0])
    return h


def _mean_pairs(y):
    """WFG2 and WFG3 distance reduction: r_nonsep over consecutive pairs, then the mean."""
    return np.mean(_r_nonsep(y.reshape(y.shape[:-1] + (-1, 2)), 2), axis=-1)


def _nonsep_groups(t):
    return _r_nonsep(t, t.shape[-1])


def wfg1(x, m=3, k=None):
    """WFG1: convex and mixed front, flat and polynomial bias. Default k = 2 (m - 1), l = 20.

    The shift step sets values below 1e-12 to 0, so the optimal distance parameters
    x_i = 0.7 i reach the front despite rounding (see the note in ``_s_linear``).
    """
    y, k = _wfg_split(x, m, k)
    y[..., k:] = _b_flat(_s_linear(y[..., k:]))
    y = y**0.02
    w = 2.0 * np.arange(1, y.shape[-1] + 1)
    g = k // (m - 1)
    pos = np.stack([_r_sum(y[..., i * g:(i + 1) * g], w[i * g:(i + 1) * g]) for i in range(m - 1)], axis=-1)
    t = np.concatenate([pos, _r_sum(y[..., k:], w[k:])[..., None]], axis=-1)
    return _wfg_f(t, False, _convex_mixed)


def wfg2(x, m=3, k=None):
    """WFG2: convex and disconnected front, nonseparable distance. l must be even."""
    y, k = _wfg_split(x, m, k, even=True)
    y[..., k:] = _s_linear(y[..., k:])
    t = _reduce(y, m, k, lambda tp: np.mean(tp, axis=-1), _mean_pairs)
    return _wfg_f(t, False, _convex_disc)


def wfg3(x, m=3, k=None):
    """WFG3: linear degenerate front, nonseparable distance. l must be even."""
    y, k = _wfg_split(x, m, k, even=True)
    y[..., k:] = _s_linear(y[..., k:])
    t = _reduce(y, m, k, lambda tp: np.mean(tp, axis=-1), _mean_pairs)
    return _wfg_f(t, True, _linear)


def _concave_wfg(y, m, k, pos=None, dist=None):
    pos = pos or (lambda tp: np.mean(tp, axis=-1))
    dist = dist or (lambda td: np.mean(td, axis=-1))
    return _wfg_f(_reduce(y, m, k, pos, dist), False, _concave)


def wfg4(x, m=3, k=None):
    """WFG4: concave front, multimodal shift on all variables."""
    y, k = _wfg_split(x, m, k)
    return _concave_wfg(_s_multi(y, 30, 10, 0.35), m, k)


def wfg5(x, m=3, k=None):
    """WFG5: concave front, deceptive shift on all variables."""
    y, k = _wfg_split(x, m, k)
    return _concave_wfg(_s_decept(y), m, k)


def wfg6(x, m=3, k=None):
    """WFG6: concave front, nonseparable reductions."""
    y, k = _wfg_split(x, m, k)
    y[..., k:] = _s_linear(y[..., k:])
    return _concave_wfg(y, m, k, _nonsep_groups, _nonsep_groups)


def wfg7(x, m=3, k=None):
    """WFG7: concave front, position parameters biased by the distance parameters."""
    y, k = _wfg_split(x, m, k)
    y[..., :k] = _b_param(y[..., :k], _mean_after(y)[..., :k])
    y[..., k:] = _s_linear(y[..., k:])
    return _concave_wfg(y, m, k)


def wfg8(x, m=3, k=None):
    """WFG8: concave front, distance parameters biased by the earlier parameters."""
    y, k = _wfg_split(x, m, k)
    y[..., k:] = _s_linear(_b_param(y[..., k:], _mean_before(y, k)))
    return _concave_wfg(y, m, k)


def wfg9(x, m=3, k=None):
    """WFG9: concave front, parameter dependent bias, deceptive and multimodal, nonseparable."""
    y, k = _wfg_split(x, m, k)
    y[..., :-1] = _b_param(y[..., :-1], _mean_after(y))
    y[..., :k] = _s_decept(y[..., :k])
    y[..., k:] = _s_multi(y[..., k:], 30, 95, 0.35)
    return _concave_wfg(y, m, k, _nonsep_groups, _nonsep_groups)


# Pareto fronts

def das_dennis(n_points, m):
    """Uniform points on the unit simplex (sum 1): the largest lattice with at most n_points.

    The lattice with H divisions has comb(H + m - 1, m - 1) points with coordinates j / H.
    Das and Dennis (1998).
    """
    if m < 1 or n_points < 1:
        raise ValueError("das_dennis needs n_points >= 1 and m >= 1")
    if m == 1:
        return np.ones((1, 1))
    h = 1
    while comb(h + m, m - 1) <= n_points:
        h += 1
    # stars and bars: the m - 1 bar positions among h + m - 1 slots
    bars = np.array(list(combinations(range(h + m - 1), m - 1)))
    edges = np.column_stack([np.full(len(bars), -1), bars, np.full(len(bars), h + m - 1)])
    return (np.diff(edges, axis=1) - 1) / h


def nondominated(F):
    """Boolean mask of the rows of F (minimized) that no other row dominates. Duplicates are kept."""
    F = np.asarray(F, dtype=float)
    keep = np.ones(len(F), dtype=bool)
    if len(F) > 1 and F.shape[1] == 2:
        # sort by f1, then f2: a row is dominated by an earlier row with a smaller f2, or with
        # the same f2 and a smaller f1
        order = np.lexsort((F[:, 1], F[:, 0]))
        f1, f2 = F[order, 0], F[order, 1]
        best = np.minimum.accumulate(f2)
        record = np.r_[True, f2[1:] < best[:-1]]
        first = np.maximum.accumulate(np.where(record, np.arange(len(f2)), 0))
        prev, at = best[:-1], f1[first[:-1]]
        dominated = (prev < f2[1:]) | ((prev == f2[1:]) & (at < f1[1:]))
        keep[order[1:]] = ~dominated
        return keep
    step = max(1, 2_000_000 // max(1, len(F) * F.shape[1]))
    for s in range(0, len(F), step):
        a = F[s:s + step, None, :]
        dominated = np.all(F[None] <= a, axis=-1) & np.any(F[None] < a, axis=-1)
        keep[s:s + step] = ~dominated.any(axis=1)
    return keep


def _first_records(values, larger=False):
    """Mask of entries strictly better than all earlier ones (a 1D nondominated sweep)."""
    best = np.maximum.accumulate(values) if larger else np.minimum.accumulate(values)
    prev = np.r_[-np.inf if larger else np.inf, best[:-1]]
    return values > prev if larger else values < prev


def _even(F, n_points):
    """n_points rows of F spread evenly by index (F has more rows)."""
    if len(F) <= n_points:
        return F
    return F[np.round(np.linspace(0, len(F) - 1, n_points)).astype(int)]


def _even_arc(F, n_points):
    """About n_points of a dense 2D curve sample F, evenly spaced in arc length.

    Dominated points are removed first, and the jumps across the gaps they leave count as
    zero length.
    """
    keep = nondominated(F)
    ds = np.linalg.norm(np.diff(F, axis=0), axis=1)[keep[1:] & keep[:-1]]
    idx = np.flatnonzero(keep)
    seg = np.zeros(len(idx) - 1)
    seg[np.diff(idx) == 1] = ds
    c = np.r_[0, np.cumsum(seg)]
    pick = np.searchsorted(c, np.linspace(0, c[-1], n_points)).clip(0, len(idx) - 1)
    return F[idx[np.unique(pick)]]


def _closest_to_directions(F, n_points, scale):
    """For each Das and Dennis direction, the row of F (divided by scale) closest to its ray."""
    G = F / scale
    w = das_dennis(n_points, F.shape[1])
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    sq = np.sum(G**2, axis=1)
    best, pick = np.full(len(w), np.inf), np.zeros(len(w), dtype=int)
    step = max(1, 4_000_000 // len(w))
    for s in range(0, len(G), step):
        # squared distance from a point to the ray along w: |g|^2 - (g . w)^2
        d = sq[None, s:s + step] - (w @ G[s:s + step].T) ** 2
        j = np.argmin(d, axis=1)
        dj = d[np.arange(len(w)), j]
        better = dj < best
        best[better], pick[better] = dj[better], s + j[better]
    return np.unique(F[np.unique(pick)], axis=0)


ZDT6_X1 = float(np.arctan(9 * PI) / (6 * PI))  # f1 is smallest here: tan(6 pi x) = 9 pi
ZDT6_F1 = float(1 - np.exp(-4 * ZDT6_X1) * np.sin(6 * PI * ZDT6_X1) ** 6)  # 0.2807753...


def _front_zdt(name, n):
    if name == "zdt5":
        f1 = np.arange(1.0, 32.0)
        return np.column_stack([f1, 10 / f1])
    if name == "zdt3":
        f1 = np.linspace(0, 1, 100 * n + 1)
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * PI * f1)
        keep = _first_records(f2)
        return _even(np.column_stack([f1[keep], f2[keep]]), n)
    f1 = np.linspace(ZDT6_F1 if name == "zdt6" else 0, 1, n)
    f2 = 1 - f1**2 if name in ("zdt2", "zdt6") else 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def _front_dtlz(name, n, m):
    if name == "dtlz1":
        return das_dennis(n, m) / 2
    if name in ("dtlz2", "dtlz3", "dtlz4"):
        w = das_dennis(n, m)
        return w / np.linalg.norm(w, axis=1, keepdims=True)
    if name in ("dtlz5", "dtlz6"):
        theta = np.full((n, m - 1), PI / 4)
        theta[:, 0] = np.linspace(0, PI / 2, n)
        return _cos_sin(theta)
    # dtlz7: separable, so the front is the product of the 1D nondominated parts
    f = np.linspace(0, 1, 20001)
    f = f[_first_records(_dtlz7_t(f), larger=True)]
    q = max(2, int(np.floor(n ** (1 / (m - 1)) + 1e-9)))
    f = _even(f, q)
    grid = np.stack([a.ravel() for a in np.meshgrid(*[f] * (m - 1), indexing="ij")], axis=-1)
    return np.column_stack([grid, 2 * (m - np.sum(_dtlz7_t(grid), axis=1))])


def _front_wfg(name, n, m):
    s = 2 * np.arange(1, m + 1)
    if name in ("wfg1", "wfg2"):
        # The whole WFG1 surface is nondominated. On WFG2 a point is nondominated exactly when
        # its first parameter is nondominated for two objectives (h_M depends on it alone, and
        # it scales the other objectives by one common factor). Dense candidates on the
        # surface, then the one closest to each reference direction for an even spread.
        shape = _convex_mixed if name == "wfg1" else _convex_disc
        if m == 2:
            return _even_arc(s * shape(np.linspace(0, 1, 100 * n)[:, None]), n)
        q = max(2, int(round((10 * n) ** (1 / (m - 1)))))
        p1 = np.linspace(0, 1, q)
        if name == "wfg2":
            p1 = p1[nondominated(_convex_disc(p1[:, None]))]
        axes = np.meshgrid(p1, *[np.linspace(0, 1, q)] * (m - 2), indexing="ij")
        p = np.stack([a.ravel() for a in axes], axis=-1)
        return _closest_to_directions(s * shape(p), n, s)
    if name == "wfg3":
        p = np.full((n, m - 1), 0.5)
        p[:, 0] = np.linspace(0, 1, n)
        return s * _linear(p)
    w = das_dennis(n, m)
    return s * w / np.linalg.norm(w, axis=1, keepdims=True)


@dataclass(frozen=True)
class MultiBenchmark:
    """Metadata of a multi-objective benchmark.

    ``n_obj`` is 2 for ZDT and None for the scalable suites. ``default_dim(m)`` gives the
    usual dimension for m objectives, and ``bounds(d)`` the lower and upper bounds.
    ``front`` describes the shape of the true Pareto front. All objectives are minimized.
    """

    fn: Callable
    suite: str
    default_dim: Callable
    lower: Callable
    upper: Callable
    front: str
    n_obj: int | None = None
    default_m: int = 2
    binary: bool = False
    sense: str = "min"

    def bounds(self, d):
        return np.asarray(self.lower(d), dtype=float), np.asarray(self.upper(d), dtype=float)


def _const(v):
    return lambda d: np.full(d, float(v))


def _zdt4_lower(d):
    return np.r_[0.0, np.full(d - 1, -5.0)]


def _zdt4_upper(d):
    return np.r_[1.0, np.full(d - 1, 5.0)]


def _wfg_upper(d):
    return 2.0 * np.arange(1, d + 1)


_ZERO, _ONE = _const(0), _const(1)


def _zdt(fn, d, front, lower=_ZERO, upper=_ONE, binary=False):
    return MultiBenchmark(fn, "ZDT", lambda m=2: d, lower, upper, front, n_obj=2, binary=binary)


def _dtlz(fn, k, front):
    return MultiBenchmark(fn, "DTLZ", lambda m=3: m - 1 + k, _ZERO, _ONE, front, default_m=3)


def _wfg(fn, front):
    return MultiBenchmark(fn, "WFG", lambda m=3: 2 * (m - 1) + 20, _ZERO, _wfg_upper, front, default_m=3)


REGISTRY = {
    "zdt1": _zdt(zdt1, 30, "convex"),
    "zdt2": _zdt(zdt2, 30, "concave"),
    "zdt3": _zdt(zdt3, 30, "disconnected"),
    "zdt4": _zdt(zdt4, 10, "convex", _zdt4_lower, _zdt4_upper),
    "zdt5": _zdt(zdt5, 80, "discrete", binary=True),
    "zdt6": _zdt(zdt6, 10, "concave"),
    "dtlz1": _dtlz(dtlz1, 5, "linear"),
    "dtlz2": _dtlz(dtlz2, 10, "concave"),
    "dtlz3": _dtlz(dtlz3, 10, "concave"),
    "dtlz4": _dtlz(dtlz4, 10, "concave"),
    "dtlz5": _dtlz(dtlz5, 10, "degenerate"),
    "dtlz6": _dtlz(dtlz6, 10, "degenerate"),
    "dtlz7": _dtlz(dtlz7, 20, "disconnected"),
    "wfg1": _wfg(wfg1, "convex, mixed"),
    "wfg2": _wfg(wfg2, "convex, disconnected"),
    "wfg3": _wfg(wfg3, "linear, degenerate"),
    "wfg4": _wfg(wfg4, "concave"),
    "wfg5": _wfg(wfg5, "concave"),
    "wfg6": _wfg(wfg6, "concave"),
    "wfg7": _wfg(wfg7, "concave"),
    "wfg8": _wfg(wfg8, "concave"),
    "wfg9": _wfg(wfg9, "concave"),
}


def pareto_front(name, n_points=1000, m=None):
    """Points on the true Pareto front of a benchmark, shape (n, m).

    DTLZ1 to DTLZ4 and WFG4 to WFG9 use the Das and Dennis lattice (the largest with at
    most n_points points), projected on the front. ZDT and the degenerate fronts (DTLZ5,
    DTLZ6, WFG3) have exactly n_points points, evenly spaced in the first parameter. ZDT5 has
    its 31 discrete points. DTLZ7 uses a grid in the front objectives with at
    most n_points points. WFG1 and WFG2 take, for each Das and Dennis direction, the point of
    a dense sample of the front closest to it, so they have at most n_points points (fewer on
    the disconnected WFG2 front, where several directions share a point at the edge of a gap).
    """
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}; known: {', '.join(REGISTRY)}")
    b = REGISTRY[key]
    if b.n_obj and m not in (None, b.n_obj):
        raise ValueError(f"{name} has {b.n_obj} objectives")
    m = b.n_obj or (b.default_m if m is None else int(m))
    if m < 2 or n_points < 2:
        raise ValueError("pareto_front needs m >= 2 and n_points >= 2")
    if b.suite == "ZDT":
        return _front_zdt(key, n_points)
    if b.suite == "DTLZ":
        return _front_dtlz(key, n_points, m)
    return _front_wfg(key, n_points, m)
