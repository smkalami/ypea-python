import numpy as np
import pytest

import ypea

ALL_VARS = [
    ypea.Real("a", shape=(2, 3), lb=-5, ub=[1, 2, 3]),
    ypea.Integer("b", shape=4, lb=-3, ub=6),
    ypea.Binary("c", shape=(3, 2)),
    ypea.Permutation("d", n=7),
    ypea.Permutation("e", n=5, count=3),
    ypea.Selection("f", n=10, k=4),
    ypea.Selection("g", n=6, k=2, count=2),
    ypea.BinarySelection("h", n=8, k=3),
    ypea.Partition("i", n=9, k=3),
    ypea.Partition("j", n=5, k=2, count=2),
    ypea.BinaryPartition("k", n=6, k=3),
    ypea.FixedSum("l", shape=5, target=10),
    ypea.FixedSum("m", shape=(3, 4), target=[1, 2, 3], axis=1),
    ypea.FixedSum("n", shape=(3, 4), target=2, axis=0),
]


def same(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


@pytest.mark.parametrize("var", ALL_VARS, ids=lambda v: type(v).__name__ + "_" + v.name)
def test_decode_encode_round_trip(var):
    rng = np.random.default_rng(0)
    codes = rng.random((20, var.code_size))
    values = var.decode(codes)
    again = var.decode(var.encode(values))
    assert same(values, again)


@pytest.mark.parametrize("var", ALL_VARS, ids=lambda v: type(v).__name__ + "_" + v.name)
@pytest.mark.parametrize("edge", [0.0, 1.0])
def test_edge_codes_decode_to_valid_values(var, edge):
    codes = np.full((1, var.code_size), edge)
    value = var.decode(codes)[0]
    if isinstance(var, ypea.Real):
        assert np.all((value >= var.lb) & (value <= var.ub))
    if isinstance(var, ypea.Integer):
        assert np.all((value >= var.lb) & (value <= var.ub))
    if isinstance(var, ypea.FixedSum):
        assert np.all(np.isfinite(value))


def test_integer_code_one_is_upper_bound():
    # MATLAB decoded a code of exactly 1 to ub + 1
    v = ypea.Integer("x", shape=3, lb=0, ub=4)
    assert v.decode(np.ones((1, 3)))[0].tolist() == [4, 4, 4]
    counts = np.bincount(v.decode(np.linspace(0, 1, 5001)[:, None].repeat(3, 1))[:, 0])
    assert len(counts) == 5 and counts.min() > 900  # every value about equally likely


def test_fixed_sum_targets_and_zero_codes():
    v = ypea.FixedSum("t", shape=(3, 4), target=[1, 2, 3], axis=1)
    x = v.decode(np.random.default_rng(1).random((5, 12)))
    assert np.allclose(x.sum(axis=2), [1, 2, 3])
    z = v.decode(np.zeros((1, 12)))[0]  # MATLAB gave NaN here
    assert np.allclose(z, np.array([1, 2, 3])[:, None] / 4)
    c = ypea.FixedSum("c", shape=(3, 4), target=[5, 6, 7, 8], axis=0)
    assert np.allclose(c.decode(np.random.default_rng(2).random((2, 12))).sum(axis=1), [5, 6, 7, 8])


def test_permutation_selection_partition_values():
    rng = np.random.default_rng(3)
    p = ypea.Permutation("p", n=6, count=2).decode(rng.random((4, 12)))
    assert p.shape == (4, 2, 6) and all(sorted(r) == list(range(6)) for r in p.reshape(-1, 6))
    s = ypea.Selection("s", n=10, k=3).decode(rng.random((4, 10)))
    assert s.shape == (4, 3) and all(len(set(r)) == 3 for r in s)
    b = ypea.BinarySelection("b", n=10, k=3).decode(rng.random((4, 10)))
    assert (b.sum(axis=1) == 3).all()
    parts = ypea.Partition("q", n=9, k=3).decode(rng.random((4, 11)))
    for groups in parts:
        assert len(groups) == 3 and sorted(np.concatenate(groups)) == list(range(9))
    m = ypea.BinaryPartition("m", n=9, k=3).decode(rng.random((4, 11)))
    assert m.shape == (4, 3, 9) and (m.sum(axis=1) == 1).all()


def test_real_shape_from_bounds():
    v = ypea.Real("z", lb=[-1, 3, -10], ub=[1, 5, 2])
    assert v.shape == (3,) and v.code_size == 3


def test_invalid_variables():
    with pytest.raises(ValueError):
        ypea.Real("x", lb=1, ub=0)
    with pytest.raises(ValueError):
        ypea.Real("not a name")
    with pytest.raises(ValueError):
        ypea.Selection("s", n=3, k=5)
