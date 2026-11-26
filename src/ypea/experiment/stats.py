"""Nonparametric tests for comparing algorithms, following Derrac et al. (2011).

* ``rank_sum``: Wilcoxon rank sum test (the Mann Whitney U test) of two samples of runs on
  one problem. Needs SciPy.
* ``friedman``: Friedman average ranks over problems, the Friedman statistic with the tie
  correction, and the Iman and Davenport statistic. Ranks need only NumPy, p values SciPy.
* ``posthoc``: the z test of each algorithm against a control (Derrac et al. 2011, eq. 12)
  with Holm adjusted p values. NumPy only.

Derrac, J., Garcia, S., Molina, D. and Herrera, F. (2011). A practical tutorial on the use
of nonparametric statistical tests as a methodology for comparing evolutionary and swarm
intelligence algorithms. Swarm and Evolutionary Computation, 1(1), 3 to 18.
"""

from __future__ import annotations

import math

import numpy as np

STATS_HINT = "statistical tests need SciPy: pip install ypea[stats]"


def scipy_stats():
    """``scipy.stats``, or None when SciPy is not installed."""
    try:
        from scipy import stats
    except ImportError:
        return None
    return stats


def rankdata(a) -> np.ndarray:
    """Ranks from 1 (smallest), ties get their average rank."""
    _, inv, counts = np.unique(np.asarray(a, dtype=float), return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts - 1) / 2
    return avg[inv]


def rank_sum(a, b, alpha=0.05, smaller_is_better=True):
    """Compare sample ``a`` with sample ``b``. Returns (mark, p value).

    The mark is "+" when ``a`` is significantly better than ``b``, "-" when it is
    significantly worse, and "=" otherwise. Raises ImportError without SciPy.
    """
    st = scipy_stats()
    if st is None:
        raise ImportError(STATS_HINT)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(np.r_[a, b]) == 0:
        return "=", 1.0  # all values equal: the test is undefined
    res = st.mannwhitneyu(a, b, alternative="two-sided")
    p = float(res.pvalue)
    if not p < alpha:
        return "=", p
    a_lower = res.statistic < len(a) * len(b) / 2  # a tends to have the smaller values
    return ("+" if a_lower == smaller_is_better else "-"), p


def friedman(values, smaller_is_better=True) -> dict:
    """Friedman test. ``values`` has one row per problem and one column per algorithm.

    Returns a dict with ``ranks`` (average rank of each algorithm, 1 is best), ``chi2`` and
    ``p`` (Friedman, with the tie correction as in ``scipy.stats.friedmanchisquare``),
    ``f`` and ``p_f`` (Iman and Davenport). The p values are None without SciPy or with
    fewer than two problems or algorithms. Missing values (NaN) rank last.
    """
    v = np.asarray(values, dtype=float)
    n, k = v.shape
    keys = np.where(np.isnan(v), np.inf, v if smaller_is_better else -v)
    r = np.array([rankdata(row) for row in keys])
    ranks = r.mean(axis=0)
    out = {"ranks": ranks, "n": n, "k": k, "chi2": None, "p": None, "f": None, "p_f": None}
    if n < 2 or k < 2:
        return out
    ties = sum(float(np.sum(c**3 - c)) for c in (np.unique(row, return_counts=True)[1] for row in keys))
    c = 1 - ties / (n * k * (k * k - 1))
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(ranks**2) - k * (k + 1) ** 2 / 4)
    chi2 = float(chi2 / c) if c > 0 else 0.0
    out["chi2"] = chi2
    denom = n * (k - 1) - chi2
    out["f"] = (n - 1) * chi2 / denom if denom > 0 else math.inf
    st = scipy_stats()
    if st is not None:
        out["p"] = float(st.chi2.sf(chi2, k - 1))
        out["p_f"] = float(st.f.sf(out["f"], k - 1, (k - 1) * (n - 1))) if denom > 0 else 0.0
    return out


def posthoc(ranks, n, control) -> dict:
    """Each algorithm against the control (an index into ``ranks``) after a Friedman test.

    ``z = (R_i - R_control) / sqrt(k (k + 1) / (6 n))`` (Derrac et al. 2011, eq. 12), with
    two sided normal p values and Holm adjusted p values,
    ``APV_i = min(1, max over j <= i of (k - j) p_j)`` for the sorted p values.
    Returns dicts from algorithm index to z, p and Holm APV (the control is left out).
    """
    ranks = np.asarray(ranks, dtype=float)
    k = len(ranks)
    se = math.sqrt(k * (k + 1) / (6 * n))
    others = [i for i in range(k) if i != control]
    z = {i: float((ranks[i] - ranks[control]) / se) for i in others}
    p = {i: math.erfc(abs(z[i]) / math.sqrt(2)) for i in others}
    holm, running = {}, 0.0
    for j, i in enumerate(sorted(others, key=lambda i: p[i]), start=1):
        running = max(running, (k - j) * p[i])
        holm[i] = min(1.0, running)
    return {"z": z, "p": p, "holm": holm}
