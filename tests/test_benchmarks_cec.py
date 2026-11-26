"""CEC 2017 and CEC 2022 suites: optimum values, batch evaluation, pickling, and
checks against formulas written out by hand from the technical reports."""

import importlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

import ypea
from ypea.benchmarks import _cec_common as cec
from ypea.benchmarks.cec2017 import cec2017, cec2017_function
from ypea.benchmarks.cec2022 import cec2022, cec2022_function

# import_module keeps the modules even if the package exports functions of the same name
c17 = importlib.import_module("ypea.benchmarks.cec2017")
c22 = importlib.import_module("ypea.benchmarks.cec2022")

REFERENCE = Path(__file__).parent / "data" / "cec" / "reference_values.json"
SUITES = {"cec2017": (c17, cec2017_function), "cec2022": (c22, cec2022_function)}


def _cases():
    for suite, (mod, _) in SUITES.items():
        for d in mod.DIMS:
            for f in mod.TABLE:
                if cec.supported(suite, mod.TABLE, f, d, mod.DIMS, mod._undefined):
                    yield suite, f, d


CASES = list(_cases())


def _problem(suite, f, d):
    return (c17.cec2017_problem if suite == "cec2017" else c22.cec2022_problem)(f, d)


@pytest.mark.parametrize(("suite", "f", "d"), CASES)
def test_value_at_optimum(suite, f, d):
    p = _problem(suite, f, d)
    fn = p.objective.function
    assert p.optimum == (100.0 * f if suite == "cec2017" else c22.BIAS[f])
    assert fn(p.info["x_opt"]) == pytest.approx(p.optimum, rel=1e-8, abs=0)
    o = fn.shift[0] if fn.shift.ndim == 2 else fn.shift
    if (suite, f) != ("cec2017", 9):
        np.testing.assert_array_equal(p.info["x_opt"], o)
    assert np.all(np.abs(p.info["x_opt"]) <= 100)


def test_cec2017_f9_value_at_shift():
    # At the shift, z = 0 and w = 0.75, so the value does not depend on the data. The
    # official CEC 2022 code carries the same D = 10 constant in a comment of levy_func.
    fn = cec2017_function(9, 10)
    assert fn(fn.shift) - 900 == pytest.approx(1.442600987052770, rel=1e-12)


@pytest.mark.parametrize(("suite", "f"), sorted({(s, f) for s, f, d in CASES if d == 10}))
def test_batch_matches_rows(suite, f):
    fn = SUITES[suite][1](f, 10)
    X = np.random.default_rng(f).uniform(-100, 100, (7, 10))
    F = fn(X)
    assert F.shape == (7,)
    rows = np.array([fn(x) for x in X])
    np.testing.assert_allclose(F, rows, rtol=1e-12)
    assert fn(X.reshape(7, 1, 10)).shape == (7, 1)


def test_suites_and_metadata():
    probs = cec2017(10)
    assert [p.info["function"] for p in probs] == [1, *range(3, 31)]
    assert [p.info["function"] for p in cec2017(10, include_f2=True)][:2] == [1, 2]
    assert len(cec2017(20)) == 18
    assert len(cec2022(10)) == 12 and len(cec2022(2)) == 9
    p = cec2017(30, functions=[5])[0]
    assert p.name == "cec2017_f5_d30" and p.vectorized and p.optimum == 500
    assert p.info["suite"] == "cec2017" and p.info["dim"] == 30 and p.info["x_opt"].shape == (30,)
    assert "github.com/P-N-Suganthan" in p.info["source"]
    assert cec2017(10, functions=[2])[0].info["excluded"]
    for bad in [(lambda: cec2017(10, functions=[31])), (lambda: cec2017(7)), (lambda: cec2017_function(11, 20)),
                (lambda: cec2017_function(21, 2)), (lambda: cec2022_function(6, 2)), (lambda: cec2022(30))]:
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        cec2022_function(1, 10)(np.zeros(5))


def test_pickle_and_short_run():
    for p in [cec2017(10, functions=[29])[0], cec2022(10, functions=[7])[0]]:
        q = pickle.loads(pickle.dumps(p))
        x = np.full(10, 3.0)
        assert q.objective.function(x) == p.objective.function(x)
        r = ypea.LSHADE(max_nfe=2000).solve(q, seed=0)
        assert r.nfe == 2000 and r.best.obj_value >= p.optimum


# hand written formulas from the technical reports, with the C loops written out


def _raw(suite, *keys):
    return cec._read(suite, list(keys))


def _rotate(M, y):
    nx = len(y)
    return np.array([sum(y[j] * M[i * nx + j] for j in range(nx)) for i in range(nx)])


def _rastrigin(z):
    return sum(v * v - 10 * math.cos(2 * math.pi * v) + 10 for v in z)


def _rosenbrock(z):
    z = [v + 1 for v in z]
    return sum(100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2 for i in range(len(z) - 1))


def test_cec2017_f5_by_hand():
    shift, M = _raw("cec2017", "shift_data_5", "M_5_D10")
    o, M = shift[0, :10], M.ravel()
    x = np.random.default_rng(1).uniform(-100, 100, 10)
    z = _rotate(M, [(x[j] - o[j]) * 5.12 / 100 for j in range(10)])
    assert cec2017_function(5, 10)(x) == pytest.approx(_rastrigin(z) + 500, rel=1e-12)


def test_cec2017_f11_hybrid_by_hand():
    shift, M, S = _raw("cec2017", "shift_data_11", "M_11_D10", "shuffle_data_11_D10")
    o, M, S = shift[0, :10], M.ravel(), S.ravel()
    x = np.random.default_rng(2).uniform(-100, 100, 10)
    z = _rotate(M, x - o)
    y = [z[S[i] - 1] for i in range(10)]
    # sizes ceil(0.2 * 10) = 2, ceil(0.4 * 10) = 4, and the rest 4
    a, b, c = y[:2], y[2:6], y[6:]
    zak = sum(v * v for v in a) + sum(0.5 * (i + 1) * v for i, v in enumerate(a)) ** 2
    zak += sum(0.5 * (i + 1) * v for i, v in enumerate(a)) ** 4
    expected = zak + _rosenbrock([v * 0.02048 for v in b]) + _rastrigin([v * 0.0512 for v in c]) + 1100
    assert cec2017_function(11, 10)(x) == pytest.approx(expected, rel=1e-12)


def test_cec2017_f6_ignores_rotation():
    # the C code computes Schaffer F7 on the shifted point y, not on the rotated z
    fn = cec2017_function(6, 10)
    x = np.random.default_rng(3).uniform(-100, 100, 10)
    y = x - fn.shift
    s = np.sqrt(y[:-1] ** 2 + y[1:] ** 2)
    expected = (np.sum(np.sqrt(s) * (1 + np.sin(50 * s**0.2) ** 2)) / 9) ** 2 + 600
    assert fn(x) == pytest.approx(expected, rel=1e-12)


def _schwefel(z):
    total = 0.0
    for v in z:
        v += 420.9687462275036
        if abs(v) <= 500:
            total += v * math.sin(math.sqrt(abs(v)))
        elif v > 500:
            m = 500 - math.fmod(v, 500)
            total += m * math.sin(math.sqrt(m)) - (v - 500) ** 2 / 10000 / len(z)
        else:
            m = math.fmod(abs(v), 500) - 500
            total += m * math.sin(math.sqrt(abs(m))) - (v + 500) ** 2 / 10000 / len(z)
    return 418.9828872724338 * len(z) - total


def test_cec2022_f10_composition_by_hand():
    shift, M = _raw("cec2022", "shift_data_10", "M_10_D10")
    oo, M = shift[:3, :10], M.ravel()
    x = np.random.default_rng(4).uniform(-100, 100, 10)
    x[:] = oo[1] + 0.5 * (x - oo[1]) / np.linalg.norm(x - oo[1])  # near component 2, so weights matter
    g1 = _schwefel([(x[j] - oo[0, j]) * 10 for j in range(10)])  # not rotated
    g2 = _rastrigin(_rotate(M[100:200], [(x[j] - oo[1, j]) * 0.0512 for j in range(10)]))
    z = [v * 0.05 - 1 for v in _rotate(M[200:300], x - oo[2])]
    r2, s = sum(v * v for v in z), sum(z)
    g3 = abs(r2**2 - s**2) ** 0.5 + (0.5 * r2 + s) / 10 + 0.5
    d2 = [float(np.sum((x - oo[i]) ** 2)) for i in range(3)]
    w = [1 / math.sqrt(d2[i]) * math.exp(-d2[i] / (2 * 10 * delta**2)) for i, delta in enumerate((20, 10, 10))]
    expected = sum(wi / sum(w) * (g + b) for wi, g, b in zip(w, (g1, g2, g3), (0, 200, 100), strict=True))
    assert cec2022_function(10, 10)(x) == pytest.approx(expected + 2400, rel=1e-12)


def test_composition_at_shift_uses_first_component():
    fn = cec2022_function(12, 20)
    assert fn(fn.shift[0]) == 2700.0  # the weight of the first component is 1e99
    assert np.isfinite(fn(fn.shift[3]))


# official values computed with the official C code


def _reference():
    if not REFERENCE.exists():
        pytest.skip("tests/data/cec/reference_values.json is missing")
    return json.loads(REFERENCE.read_text())


def _reference_cases():
    if not REFERENCE.exists():
        return [pytest.param(None, None, None, marks=pytest.mark.skip(reason="reference file missing"))]
    ref = _reference()
    return [(s, int(d), int(f)) for s in SUITES for d in ref.get(s, {}) for f in ref[s][d]]


@pytest.mark.parametrize(("suite", "d", "f"), _reference_cases())
def test_against_official_values(suite, d, f):
    if (suite, f) == ("cec2017", 2):
        pytest.skip("the official code no longer evaluates F2, so its stored values are those of F1")
    case = _reference()[suite][str(d)][str(f)]
    F = SUITES[suite][1](f, d)(np.asarray(case["x"], dtype=float))
    np.testing.assert_allclose(F, case["f"], rtol=1e-10, atol=0)
