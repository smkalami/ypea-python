"""ZDT, DTLZ and WFG benchmarks, their Pareto fronts, and the quality indicators.

The pymoo reference values in data/multi/pymoo_reference.npz come from pymoo 0.6.2 (seeded
random points, the bound corners, and optimal WFG solutions). pymoo is not needed here.
"""

from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import pytest

from ypea import indicators as ind
from ypea.benchmarks import multi
from ypea.benchmarks.multi import REGISTRY, das_dennis, nondominated, pareto_front

REF = np.load(Path(__file__).parent / "data" / "multi" / "pymoo_reference.npz")
KEYS = sorted({k[:-2] for k in REF.files if k.endswith("_x")})


def cases(names, ms=(2, 3, 5)):
    return [(n, m) for n in names for m in ((REGISTRY[n].n_obj,) if REGISTRY[n].n_obj else ms)]


def evaluate(name, x, m=None):
    b = REGISTRY[name]
    if b.suite == "ZDT":
        return b.fn(x)
    if b.suite == "DTLZ":
        return b.fn(x, m=m)
    return b.fn(x, m=m, k=x.shape[-1] - 20)


def split(key):
    name, _, m = key.partition("_m")
    return name, int(m) if m else None


@pytest.mark.parametrize("key", KEYS)
def test_values_match_pymoo(key):
    name, m = split(key)
    f = evaluate(name, REF[key + "_x"], m)
    assert np.allclose(f, REF[key + "_f"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("key", [k for k in KEYS if k.startswith("wfg") and not k.startswith("wfg1_")])
def test_wfg_optimal_solutions_match_pymoo(key):
    name, m = split(key)
    assert np.allclose(evaluate(name, REF[key + "_ps"], m), REF[key + "_psf"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name,m", cases(REGISTRY))
def test_shapes_bounds_and_no_warnings(name, m):
    b = REGISTRY[name]
    d = b.default_dim(m)
    lo, hi = b.bounds(d)
    assert lo.shape == hi.shape == (d,) and np.all(lo < hi)
    rng = np.random.default_rng(0)
    x = lo + rng.random((7, d)) * (hi - lo)
    if b.binary:
        x = np.round(x)
    x = np.vstack([x, lo, hi])
    f = evaluate(name, x, m) if b.n_obj is None else b.fn(x)
    assert f.shape == (9, m) and np.all(np.isfinite(f))
    one = evaluate(name, x[0], m) if b.n_obj is None else b.fn(x[0])
    assert one.shape == (m,) and np.allclose(one, f[0])
    batch = evaluate(name, x.reshape(3, 3, d), m) if b.n_obj is None else b.fn(x.reshape(3, 3, d))
    assert np.allclose(batch.reshape(9, m), f)


def test_registry_defaults():
    assert {n: REGISTRY[n].default_dim() for n in ("zdt1", "zdt4", "zdt5", "zdt6")} == {
        "zdt1": 30, "zdt4": 10, "zdt5": 80, "zdt6": 10}
    assert [REGISTRY[f"dtlz{i}"].default_dim(3) for i in range(1, 8)] == [7, 12, 12, 12, 12, 12, 22]
    assert REGISTRY["wfg1"].default_dim(3) == 24 and REGISTRY["wfg1"].default_dim(5) == 28
    assert all(b.sense == "min" for b in REGISTRY.values())
    lo, hi = REGISTRY["zdt4"].bounds(10)
    assert lo[0] == 0 and hi[0] == 1 and np.all(lo[1:] == -5) and np.all(hi[1:] == 5)
    assert np.array_equal(REGISTRY["wfg4"].bounds(3)[1], [2, 4, 6])


def test_known_values():
    assert np.allclose(multi.zdt5(np.zeros(80)), [1, 20])
    assert np.allclose(multi.zdt5(np.ones(80)), [31, 10 / 31])
    assert np.allclose(multi.zdt1(np.r_[0.25, np.zeros(29)]), [0.25, 0.5])
    assert np.allclose(multi.dtlz1(np.full(7, 0.5)), [0.125, 0.125, 0.25])
    assert np.allclose(multi.dtlz2(np.r_[0, 0, np.full(10, 0.5)]), [1, 0, 0])
    assert np.allclose(multi.dtlz7(np.zeros(22)), [0, 0, 6])
    with pytest.raises(ValueError):
        multi.zdt5(np.zeros(33))
    with pytest.raises(ValueError):
        multi.wfg1(np.ones(24), m=3, k=3)  # k must be a multiple of m - 1
    with pytest.raises(ValueError):
        multi.wfg2(np.ones(25), m=3, k=4)  # an odd number of distance parameters


def optimal_x(name, m, n, rng):
    """Pareto optimal solutions: free position parameters, optimal distance parameters."""
    b = REGISTRY[name]
    d = b.default_dim(m)
    lo, hi = b.bounds(d)
    x = lo + rng.random((n, d)) * (hi - lo)
    x[:, 0] = np.linspace(0, 1, n) * hi[0]
    if b.suite == "ZDT":
        x[:, 1:] = 0
    elif b.suite == "DTLZ":
        x[:, m - 1:] = 0 if name in ("dtlz6", "dtlz7") else 0.5
    else:
        x[:, 2 * (m - 1):] = 0.35 * hi[2 * (m - 1):]
    return x


def on_front(name, f, m):
    """Distance of f from the analytic front (0 when f is on it)."""
    s = 2.0 * np.arange(1, m + 1)
    if name in ("zdt1", "zdt4"):
        return f[:, 1] - (1 - np.sqrt(f[:, 0]))
    if name in ("zdt2", "zdt6"):
        return f[:, 1] - (1 - f[:, 0] ** 2)
    if name == "zdt3":
        return f[:, 1] - (1 - np.sqrt(f[:, 0]) - f[:, 0] * np.sin(10 * np.pi * f[:, 0]))
    if name == "dtlz1":
        return f.sum(axis=1) - 0.5
    if name.startswith("dtlz") and name != "dtlz7":
        return np.sum(f**2, axis=1) - 1
    if name == "dtlz7":
        t = f[:, :-1] / 2 * (1 + np.sin(3 * np.pi * f[:, :-1]))
        return f[:, -1] - 2 * (m - t.sum(axis=1))
    if name == "wfg3":
        return np.sum(f / s, axis=1) - 1
    return np.sum((f / s) ** 2, axis=1) - 1


ANALYTIC = [n for n in REGISTRY if n not in ("zdt5", "wfg1", "wfg2")]


@pytest.mark.parametrize("name,m", cases(ANALYTIC))
def test_fronts_lie_on_the_analytic_front(name, m):
    front = pareto_front(name, 300, m=m)
    assert front.shape[1] == m and len(front) <= 300
    assert np.allclose(on_front(name, front, m), 0, atol=1e-12)
    assert nondominated(front).all()


@pytest.mark.parametrize("name,m", cases([n for n in ANALYTIC if n not in ("wfg8", "wfg9")] + ["wfg1"], (2, 3)))
def test_pareto_optimal_solutions_reach_the_front(name, m):
    f = evaluate(name, optimal_x(name, m, 50, np.random.default_rng(3)), m)
    if name == "wfg1":
        # f is on the front surface: close to a dense sample of it
        front = pareto_front(name, 4000 if m == 2 else 2000, m=m)
        assert np.max(ind._min_dist(f, front)) < (0.005 if m == 2 else 0.3)
    else:  # for ZDT3 and DTLZ7 this includes the dominated parts of the curve
        assert np.allclose(on_front(name, f, m), 0, atol=1e-12)


@pytest.mark.parametrize("key", [k for k in KEYS if k.startswith("wfg") and k[3] in "456789"])
def test_pymoo_optimal_wfg_solutions_are_on_the_front(key):
    name, m = split(key)
    assert np.allclose(on_front(name, evaluate(name, REF[key + "_ps"], m), m), 0, atol=1e-12)


def test_pymoo_fronts_are_covered():
    """Every point of pymoo's front lies near ours, or is dominated by ours.

    pymoo's DTLZ7 and WFG2 reference fronts keep dominated points, so far points must be
    dominated by ours. Its WFG1 front is shifted by 0.0695 in every objective (rounding at the
    optimal distance parameters), and its ZDT5 front is the normalized one, so it is skipped.
    """
    for key in [k[:-6] for k in REF.files if k.endswith("_front") and not k.startswith("ind")]:
        name, m = split(key)
        if name == "zdt5":
            continue
        ours = pareto_front(name, 4000, m=m)
        theirs = REF[key + "_front"] - (0.06946395 if name == "wfg1" else 0)
        theirs = theirs[:: max(1, len(theirs) // 300)]
        far = ind._min_dist(theirs, ours) > (0.01 if m == 2 else 0.3 if name in ("wfg1", "wfg2") else 0.1)
        dominated = np.array([np.any(np.all(ours <= p, 1) & np.any(ours < p, 1)) for p in theirs[far]])
        assert dominated.all(), key


def test_disconnected_and_degenerate_fronts():
    zdt3 = pareto_front("zdt3", 500)
    gaps = np.diff(zdt3[:, 0])
    assert np.sum(gaps > 0.05) == 4  # five parts
    assert np.isclose(zdt3[:, 0].max(), 0.8518, atol=1e-3)
    zdt6 = pareto_front("zdt6", 50)
    assert np.isclose(zdt6[0, 0], 0.2807753191, atol=1e-9)
    d7 = pareto_front("dtlz7", 400, m=3)
    assert len(d7) == 400  # 20 by 20
    f1 = np.unique(d7[:, 0])
    assert np.sum(np.diff(f1) > 0.1) == 1  # two intervals per position objective
    assert np.isclose(f1.max(), 0.8594, atol=1e-3)
    d5 = pareto_front("dtlz5", 100, m=4)
    assert np.allclose(d5[:, 0], d5[:, 1]) and np.allclose(d5[:, 2], d5[:, 1] * np.sqrt(2))  # a curve
    w3 = pareto_front("wfg3", 100, m=3)
    assert np.allclose(w3[:, 1], 2 * w3[:, 0])  # a line
    w2 = pareto_front("wfg2", 2000, m=2)
    assert nondominated(w2).all() and 1900 < len(w2) <= 2000
    w2_3 = pareto_front("wfg2", 900, m=3)
    assert nondominated(w2_3).all() and len(w2_3) < 900
    assert len(pareto_front("zdt5", 10)) == 31


def test_front_sizes_and_errors():
    assert len(pareto_front("dtlz2", 91, m=3)) == 91
    assert len(pareto_front("dtlz2", 100, m=3)) == 91
    assert len(pareto_front("zdt1", 77)) == 77
    assert len(pareto_front("wfg4", 210, m=5)) <= 210
    with pytest.raises(KeyError):
        pareto_front("zdt7", 10)
    with pytest.raises(ValueError):
        pareto_front("zdt1", 10, m=3)
    assert np.array_equal(pareto_front("DTLZ1", 10, m=2), pareto_front("dtlz1", 10, m=2))


@pytest.mark.parametrize("n,m", [(1, 2), (10, 2), (91, 3), (100, 3), (210, 5), (5, 5)])
def test_das_dennis(n, m):
    w = das_dennis(n, m)
    assert np.allclose(w.sum(axis=1), 1) and np.all(w >= 0)
    assert len(np.unique(w, axis=0)) == len(w)
    h = round(1 / np.min(w[w > 0]))
    assert len(w) == comb(h + m - 1, m - 1)
    assert len(w) <= max(n, m) and comb(h + m, m - 1) > n


def test_nondominated():
    F = np.array([[1, 2], [2, 1], [2, 2], [1, 2], [0, 3], [3, 3]], float)
    assert nondominated(F).tolist() == [True, True, False, True, True, False]
    F3 = np.array([[1, 1, 1], [0, 2, 2], [1, 1, 2], [1, 1, 1]], float)
    assert nondominated(F3).tolist() == [True, True, False, True]
    rng = np.random.default_rng(1)
    for _ in range(50):
        F = rng.integers(0, 4, (rng.integers(1, 15), 2)).astype(float)
        slow = [not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in range(len(F))) for i in range(len(F))]
        assert nondominated(F).tolist() == slow


# indicators

def brute_hv(points, ref):
    """Hypervolume by summing the grid cells (between point coordinates) that are dominated."""
    p = np.asarray(points, float)
    p = p[np.all(p < ref, axis=1)]
    if not len(p):
        return 0.0
    axes = [np.unique(np.r_[p[:, j], ref[j]]) for j in range(len(ref))]
    total = 0.0
    for idx in product(*[range(len(a) - 1) for a in axes]):
        corner = np.array([a[i] for a, i in zip(axes, idx, strict=True)])
        if np.any(np.all(p <= corner, axis=1)):
            total += np.prod([a[i + 1] - a[i] for a, i in zip(axes, idx, strict=True)])
    return total


def test_hypervolume_hand_cases():
    assert ind.hypervolume([[1, 2], [2, 1]], [3, 3]) == pytest.approx(3)
    assert ind.hypervolume([[0, 0, 0]], [1, 2, 3]) == pytest.approx(6)
    assert ind.hypervolume([[1, 0, 0], [0, 1, 0]], [2, 2, 2]) == pytest.approx(8 - 2)
    assert ind.hypervolume([[1.0]], [3.0]) == 2
    assert ind.hypervolume(np.empty((0, 2)), [1, 1]) == 0
    assert ind.hypervolume([], [1, 1]) == 0
    assert ind.hypervolume([[1, 0.5], [0.5, 2]], [1, 1]) == 0  # nothing strictly inside
    pts = [[1, 2], [2, 1]]
    assert ind.hypervolume(pts + pts + [[2.5, 2.5]], [3, 3]) == pytest.approx(3)  # duplicates, dominated


@pytest.mark.parametrize("m", [2, 3])
def test_hypervolume_exact_against_brute_force(m):
    rng = np.random.default_rng(m)
    for n in (1, 3, 8, 15):
        pts = rng.random((n, m))
        pts[0] = 1.2  # one point outside the reference box
        pts = np.round(np.vstack([pts, pts[-1:]]), 2)  # repeats and ties
        ref = np.ones(m)
        assert ind.hypervolume(pts, ref) == pytest.approx(brute_hv(pts, ref), rel=1e-12, abs=1e-15)


def test_hypervolume_monte_carlo():
    rng = np.random.default_rng(5)
    pts, ref = rng.random((8, 4)), np.full(4, 1.0)
    exact = brute_hv(pts, ref)
    est = ind.hypervolume(pts, ref, n_samples=200_000, seed=2)
    box = np.prod(ref - pts.min(axis=0))
    q = exact / box
    assert abs(est - exact) < 4 * box * np.sqrt(q * (1 - q) / 200_000)
    assert est == ind.hypervolume(pts, ref, n_samples=200_000, seed=2)
    assert est != ind.hypervolume(pts, ref, n_samples=200_000, seed=3)


@pytest.mark.parametrize("key", [k[:-6] for k in REF.files if k.startswith("hv_") and k.endswith("_value")])
def test_hypervolume_matches_pymoo(key):
    pts, ref, value = REF[key + "_points"], REF[key + "_ref"], float(REF[key + "_value"])
    if len(ref) <= 3:
        assert ind.hypervolume(pts, ref) == pytest.approx(value, rel=1e-12)
    else:
        assert ind.hypervolume(pts, ref, n_samples=400_000, seed=0) == pytest.approx(value, rel=0.01)


def test_distance_indicators():
    front = np.array([[0.0, 1.0], [1.0, 0.0]])
    pts = np.array([[0.0, 2.0], [1.0, 0.0]])
    assert ind.gd(pts, front) == pytest.approx(0.5)
    assert ind.igd(pts, front) == pytest.approx(0.5)
    assert ind.igd(front, front) == 0 and ind.gd(front, front) == 0
    better = np.array([[0.0, 0.5], [0.5, 0.0]])  # dominates the front: GD+ and IGD+ are 0
    assert ind.gd(better, front) > 0 and ind.gd_plus(better, front) == 0 and ind.igd_plus(better, front) == 0
    assert ind.gd_plus([[0.5, 1.5]], front) == pytest.approx(np.hypot(0.5, 0.5))
    assert ind.igd_plus([[0.5, 1.5]], front) == pytest.approx((np.hypot(0.5, 0.5) + 1.5) / 2)
    for f in (ind.gd, ind.igd, ind.gd_plus, ind.igd_plus):
        assert f(np.empty((0, 2)), front) == np.inf
        assert f(pts, front) == pytest.approx(getattr(ind, f.__name__)(pts.tolist(), front.tolist()))
        with pytest.raises(ValueError):
            f(pts, np.empty((0, 2)))
        with pytest.raises(ValueError):
            f(np.ones((2, 3)), front)
    assert ind.gd([0.0, 2.0], front) == pytest.approx(1.0)  # one point as a 1D array


@pytest.mark.parametrize("name", ["gd", "igd", "gd_plus", "igd_plus"])
def test_distance_indicators_match_pymoo(name):
    value = getattr(ind, name)(REF["ind_points"], REF["ind_front"])
    assert value == pytest.approx(float(REF["ind_" + name]), rel=1e-12)


def test_spacing():
    assert ind.spacing([[0, 0], [1, 0], [3, 0]]) == pytest.approx(np.sqrt(1 / 3))
    assert ind.spacing(np.column_stack([np.linspace(0, 1, 11), np.linspace(1, 0, 11)])) == pytest.approx(0, abs=1e-12)
    assert np.isnan(ind.spacing([[1, 2]])) and np.isnan(ind.spacing([]))


def test_indicators_on_a_true_front():
    front = pareto_front("dtlz2", 2000, m=3)
    assert ind.igd(front, front) == 0
    hv = ind.hypervolume(front, [1.1, 1.1, 1.1])
    # the dominated volume of the full sphere front: 1.1^3 - pi / 6
    assert hv == pytest.approx(1.1**3 - np.pi / 6, abs=0.02) and hv < 1.1**3 - np.pi / 6


def test_multi_problem_factory():
    import pickle

    import ypea
    from ypea.benchmarks import multi_problem

    p = multi_problem("dtlz2")
    assert p.name == "dtlz2_m3_d12" and p.n_obj == 3 and p.code_size == 12
    assert p.info["pareto_front"].shape == (990, 3) and p.info["suite"] == "DTLZ"
    x = np.random.default_rng(0).random((4, 12))
    assert np.array_equal(p.objective(type("S", (), {"x": x})()), multi.dtlz2(x, m=3))
    w = multi_problem("wfg4", m=2)
    assert w.name == "wfg4_m2_d22" and np.allclose(w.vars[0].ub, 2.0 * np.arange(1, 23))
    z = pickle.loads(pickle.dumps(multi_problem("zdt5")))
    assert isinstance(z.vars[0], ypea.Binary) and z.info["pareto_front"].shape == (31, 2)
    r = ypea.RandomSearch(max_nfe=100).solve(z, seed=0)
    assert r.front.obj_values[:, 0].min() >= 1  # f1 = 1 + the number of ones
    with pytest.raises(ValueError, match="2 objectives"):
        multi_problem("zdt1", m=3)
    with pytest.raises(ValueError, match="unknown"):
        multi_problem("zdt7")
    with pytest.raises(ValueError, match="even"):
        multi_problem("wfg2", dim=25)
