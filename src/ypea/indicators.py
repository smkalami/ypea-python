"""Quality indicators for sets of objective vectors (all objectives minimized).

``points`` is an approximation set with shape (n, m), for example the front of a run, and
``front`` is a sample of the true Pareto front (see ``ypea.benchmarks.multi.pareto_front``).
Distances are Euclidean. An empty approximation set gives inf for the distance indicators
and 0 for the hypervolume.

Formulas, with d(a, Z) = min over z in Z of ||a - z||:

- GD (generational distance) = mean over a in points of d(a, front). Van Veldhuizen and
  Lamont (1998) defined sqrt(sum d^2) / n. The mean (the GD_p of Schuetze et al. 2012
  with p = 1) does not shrink as n grows, and it is what pymoo computes.
- IGD (inverted generational distance) = mean over z in front of d(z, points).
  Coello Coello and Reyes Sierra (2004).
- GD+ and IGD+ use d+(a, z) = ||max(a - z, 0)||, the distance in the dominated direction
  only, so they are weakly Pareto compliant. Ishibuchi, Masuda, Tanigaki and Nojima (2015),
  "Modified distance calculation in generational distance and inverted generational
  distance", EMO 2015.
- Hypervolume = the volume of the region dominated by the points and bounded by the
  reference point. Zitzler and Thiele (1998). Exact for up to 3 objectives (a sort for 2,
  slices along the third objective for 3, as in the HSO of While, Hingston, Barone and Huband
  2006). Monte Carlo estimate above 3 objectives.
- Spacing = sqrt(sum (mean(d) - d_i)^2 / (n - 1)) with d_i the smallest L1 distance from
  point i to another point. Schott (1995), MSc thesis, MIT. 0 means evenly spaced.
"""

from __future__ import annotations

import numpy as np

from .benchmarks.multi import nondominated

__all__ = ["gd", "gd_plus", "hypervolume", "igd", "igd_plus", "spacing"]


def _set(a, name, m=None):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.empty((0, m or 0))
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or (m is not None and a.shape[1] != m):
        raise ValueError(f"{name} must have shape (n, {m or 'm'}), got {a.shape}")
    return a


def _min_dist(a, b, plus=False):
    """For each row of a, the smallest distance to the rows of b (chunked to save memory)."""
    out = np.empty(len(a))
    step = max(1, 4_000_000 // max(1, len(b) * a.shape[1]))
    for s in range(0, len(a), step):
        diff = a[s:s + step, None, :] - b[None, :, :]
        if plus:
            diff = np.maximum(diff, 0)
        out[s:s + step] = np.sqrt(np.min(np.sum(diff**2, axis=-1), axis=1))
    return out


def _pair(points, front):
    front = _set(front, "front")
    if len(front) == 0:
        raise ValueError("the reference front is empty")
    return _set(points, "points", front.shape[1]), front


def gd(points, front):
    """Generational distance: the mean distance from each point to the nearest front point."""
    points, front = _pair(points, front)
    return float(np.mean(_min_dist(points, front))) if len(points) else np.inf


def igd(points, front):
    """Inverted generational distance: the mean distance from each front point to the set."""
    points, front = _pair(points, front)
    return float(np.mean(_min_dist(front, points))) if len(points) else np.inf


def gd_plus(points, front):
    """GD+: like GD with the distance counted only where a point is worse than the front."""
    points, front = _pair(points, front)
    return float(np.mean(_min_dist(points, front, plus=True))) if len(points) else np.inf


def igd_plus(points, front):
    """IGD+: like IGD with the distance counted only where the set is worse than the front."""
    points, front = _pair(points, front)
    if not len(points):
        return np.inf
    # d+(a, z) = ||max(a - z, 0)|| for each front point z against every point a
    out = np.empty(len(front))
    step = max(1, 4_000_000 // max(1, len(points) * front.shape[1]))
    for s in range(0, len(front), step):
        diff = np.maximum(points[None, :, :] - front[s:s + step, None, :], 0)
        out[s:s + step] = np.sqrt(np.min(np.sum(diff**2, axis=-1), axis=1))
    return float(np.mean(out))


def _hv2(p, ref):
    """Exact area for two objectives (p strictly inside the reference point)."""
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    best = np.minimum.accumulate(p[:, 1])
    prev = np.r_[ref[1], best[:-1]]
    return float(np.sum((ref[0] - p[:, 0]) * np.maximum(prev - p[:, 1], 0)))


def _hv3(p, ref):
    """Exact volume for three objectives: area of the first i points times the slab depth."""
    # shortcut: O(n^2 log n), fine for a few thousand points; use a dimension sweep for more
    p = p[np.argsort(p[:, 2], kind="stable")]
    depth = np.diff(np.r_[p[:, 2], ref[2]])
    return float(sum(_hv2(p[: i + 1, :2], ref[:2]) * depth[i] for i in range(len(p)) if depth[i] > 0))


def hypervolume(points, ref, n_samples=1_000_000, seed=0):
    """Volume dominated by ``points`` and bounded by the reference point ``ref``.

    Points that do not strictly dominate ``ref`` add nothing and are ignored, and so do
    dominated and repeated points. The value is exact for 1 to 3 objectives. For 4 and
    more it is a Monte Carlo estimate from ``n_samples`` uniform samples in the box between
    the best point coordinates and ``ref``, drawn from ``numpy.random.default_rng(seed)``,
    so the same seed gives the same value. The standard error is about
    V sqrt(q (1 - q) / n_samples) for a box volume V and a dominated share q.
    """
    ref = np.asarray(ref, dtype=float).ravel()
    m = len(ref)
    p = _set(points, "points", m)
    p = p[np.all(p < ref, axis=1)]
    if len(p) == 0:
        return 0.0
    if m == 1:
        return float(ref[0] - p.min())
    p = np.unique(p, axis=0)
    p = p[nondominated(p)]
    if m == 2:
        return _hv2(p, ref)
    if m == 3:
        return _hv3(p, ref)
    lo = p.min(axis=0)
    box = float(np.prod(ref - lo))
    rng = np.random.default_rng(seed)
    hits, n_samples = 0, int(n_samples)
    step = max(1, 4_000_000 // (len(p) * m))
    for s in range(0, n_samples, step):
        u = lo + rng.random((min(step, n_samples - s), m)) * (ref - lo)
        hits += int(np.count_nonzero(np.any(np.all(p[None] <= u[:, None], axis=-1), axis=1)))
    return box * hits / n_samples


def spacing(points):
    """Schott's spacing: the spread of nearest neighbor L1 distances (0 is evenly spaced).

    Needs at least two points: fewer give NaN.
    """
    p = _set(points, "points")
    if len(p) < 2:
        return np.nan
    # shortcut: an n by n matrix, fine for fronts of a few thousand points
    d = np.sum(np.abs(p[:, None, :] - p[None, :, :]), axis=-1)
    np.fill_diagonal(d, np.inf)
    d = d.min(axis=1)
    return float(np.sqrt(np.sum((d.mean() - d) ** 2) / (len(p) - 1)))
