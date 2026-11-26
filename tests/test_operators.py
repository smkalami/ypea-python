"""Native operators for GA and SA: type invariants, round trips, and native runs."""

import pickle

import numpy as np
import pytest
from problems import mixed_obj, sphere_obj

import ypea
from ypea.operators import (
    BinarySelectionOperators,
    IntegerOperators,
    NativeOperators,
    PartitionOperators,
    PermutationOperators,
    RealOperators,
    SelectionOperators,
    cycle_crossover,
    default_operators,
    insertion,
    inversion,
    one_point_crossover,
    order_crossover,
    pmx,
    scramble,
    swap,
    uniform_crossover,
)

VARS = [
    ypea.Real("r", shape=4, lb=[-1, 0, 2, 5], ub=[1, 3, 2, 9]),
    ypea.Integer("i", shape=(2, 3), lb=-2, ub=4),
    ypea.Integer("one", lb=3, ub=3),
    ypea.Binary("b", shape=7),
    ypea.Permutation("p", n=9),
    ypea.Permutation("pc", n=6, count=3),
    ypea.Selection("s", n=8, k=3),
    ypea.Selection("s1", n=5, k=1),
    ypea.BinarySelection("bs", n=8, k=3),
    ypea.BinarySelection("bsc", n=5, k=2, count=2),
]


def valid(var, y):
    if isinstance(var, ypea.Real):
        return np.all((y >= var.lb) & (y <= var.ub))
    if isinstance(var, ypea.Integer):
        return y.dtype.kind == "i" and np.all((y >= var.lb) & (y <= var.ub))
    rows = y.reshape(-1, y.shape[-1])
    if isinstance(var, ypea.Permutation):
        return all(sorted(r) == list(range(var.n)) for r in rows)
    if isinstance(var, ypea.Selection):
        return all(len(set(r)) == var.k and r.min() >= 0 and r.max() < var.n for r in rows)
    if isinstance(var, ypea.BinarySelection):
        return np.all(rows.sum(axis=1) == var.k) and set(np.unique(rows)) <= {0, 1}
    raise AssertionError(var)


def outputs(var, rng):
    x1, x2 = (var.decode(rng.random((30, var.code_size))) for _ in range(2))
    ops = default_operators(var, rate=0.3, sigma=0.5)
    yield from ops.crossover(x1, x2, rng)
    yield ops.mutate(x1, rng)
    if isinstance(var, ypea.Integer):
        yield from uniform_crossover(x1, x2, rng)
        yield from one_point_crossover(x1, x2, rng)
    if isinstance(var, ypea.Permutation):
        every = PermutationOperators(var, crossovers=(order_crossover, pmx, cycle_crossover),
                                     mutations=(swap, inversion, insertion, scramble))
        for c in every.crossovers:
            yield from PermutationOperators(var, crossovers=(c,)).crossover(x1, x2, rng)
        for m in every.mutations:
            yield PermutationOperators(var, mutations=(m,)).mutate(x1, rng)


@pytest.mark.parametrize("var", VARS, ids=lambda v: v.name)
@pytest.mark.parametrize("rng", [0, 1, "studio"])
def test_operators_keep_invariants_and_round_trip(var, rng):
    rng = ypea.StudioRng(3) if rng == "studio" else np.random.default_rng(rng)
    for _ in range(5):
        for y in outputs(var, rng):
            assert y.shape == (30, *var.shape)
            assert valid(var, y)
            back = var.decode(var.encode(y))
            # reals round trip up to rounding, every other type exactly
            assert np.allclose(back, y) if isinstance(var, ypea.Real) else np.array_equal(back, y)


def test_operator_details():
    p1, p2 = np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([7, 6, 5, 4, 3, 2, 1, 0])
    assert order_crossover(p1, p2, 2, 5).tolist() == [6, 5, 2, 3, 4, 1, 0, 7]
    assert pmx(p1, p2, 2, 5).tolist() == [7, 6, 2, 3, 4, 5, 1, 0]
    assert cycle_crossover(np.array([0, 1, 2, 3]), np.array([1, 0, 3, 2])).tolist() == [0, 1, 3, 2]
    assert insertion(p1, 0, 3, None).tolist() == [1, 2, 3, 0, 4, 5, 6, 7]
    assert inversion(p1, 5, 2, None).tolist() == [0, 1, 5, 4, 3, 2, 6, 7]
    # creep stays within one step of the parent when sigma is small
    var = ypea.Integer("i", shape=50, lb=0, ub=100)
    x = np.full((4, 50), 50)
    y = IntegerOperators(var, rate=1.0, sigma=0.0, creep=1.0).mutate(x, np.random.default_rng(0))
    assert set(np.unique(np.abs(y - x))) == {1}
    # the defaults of each type
    kinds = {v.name: type(default_operators(v)) for v in VARS}
    assert kinds["r"] is RealOperators and kinds["b"] is IntegerOperators
    assert kinds["s"] is SelectionOperators and kinds["bs"] is BinarySelectionOperators
    assert type(default_operators(ypea.Partition("q", 5, 2))) is PartitionOperators
    assert type(default_operators(ypea.BinaryPartition("q", 5, 2))) is PartitionOperators
    assert default_operators(ypea.FixedSum("f", 3)) is None


PARTS = [
    ypea.Partition("q", n=7, k=3),
    ypea.Partition("q1", n=5, k=1),
    ypea.Partition("qc", n=6, k=2, count=2),
    ypea.BinaryPartition("bq", n=6, k=3),
    ypea.BinaryPartition("bqc", n=5, k=2, count=2),
]


def partition_rows(var, y):
    """Each sample as a tuple of rows, each row a tuple of groups (item order kept)."""
    if isinstance(var, ypea.BinaryPartition):
        rows = np.asarray(y).reshape(-1, var.rows, var.k, var.n)
        return [tuple(tuple(tuple(np.flatnonzero(g).tolist()) for g in row) for row in s) for s in rows]
    samples = y if var.count else [[s] for s in y]
    return [tuple(tuple(tuple(int(i) for i in g) for g in row) for row in s) for s in samples]


@pytest.mark.parametrize("var", PARTS, ids=lambda v: v.name)
@pytest.mark.parametrize("rng", [0, "studio"])
def test_partition_operators_keep_groups_and_round_trip(var, rng):
    rng = ypea.StudioRng(5) if rng == "studio" else np.random.default_rng(rng)
    x1, x2 = (var.decode(rng.random((30, var.code_size))) for _ in range(2))
    every = (swap, inversion, insertion, scramble)
    outs = [*default_operators(var).crossover(x1, x2, rng), default_operators(var).mutate(x1, rng)]
    for c in (order_crossover, pmx, cycle_crossover):
        outs += PartitionOperators(var, crossovers=(c,)).crossover(x1, x2, rng)
    outs += [PartitionOperators(var, mutations=(m,)).mutate(x1, rng) for m in every]
    for y in outs:
        rows = partition_rows(var, y)
        assert len(rows) == 30
        for row in (r for s in rows for r in s):
            assert len(row) == var.k and sorted(i for g in row for i in g) == list(range(var.n))
        assert partition_rows(var, var.decode(var.encode(y))) == rows


def test_partition_ordering_numbers_separators_by_group():
    var = ypea.Partition("q", n=5, k=3)
    ops = PartitionOperators(var)
    x = [[np.array([3, 0]), np.array([], int), np.array([4, 1, 2])]]
    assert ops._orders(x).tolist() == [[3, 0, 5, 6, 4, 1, 2]]
    # swapping item 0 with the first separator moves item 0 into the empty middle group
    y = ops._values(np.array([[3, 5, 0, 6, 4, 1, 2]]))
    assert partition_rows(var, y) == [(((3,), (0,), (4, 1, 2)),)]


def test_default_space_is_unchanged(sphere10):
    # values recorded from GA and SA before native operators were added (GA with its old default inflation)
    r = ypea.GA(max_nfe=3000, pop_size=30, crossover_inflation=0.1).solve(sphere10, seed=0)
    assert r.best.obj_value == 0.0012462842352094355 and len(r.history.trace_nfe) == 143
    r = ypea.SA(initial_temp=10, final_temp=1e-4, max_nfe=3000).solve(sphere10, seed=0)
    assert r.best.obj_value == 0.48971543105868015 and len(r.history.trace_nfe) == 56


NATIVE = {
    "GA": lambda **k: ypea.GA(space="native", **k),
    "SA": lambda **k: ypea.SA(space="native", initial_temp=10, final_temp=1e-3, pop_size=5, **k),
}


@pytest.mark.parametrize("name", NATIVE)
def test_native_runs_on_every_variable_type(name, mixed):
    alg = NATIVE[name](max_nfe=1500)
    r = alg.solve(mixed, seed=0)
    sol = r.best.solution
    assert sorted(sol.p) == list(range(5)) and sum(len(g) for g in sol.q) == 6
    assert np.isclose(sol.fs.sum(), 1) and r.best.obj_value == mixed_obj(sol)
    a, b = alg.solve(mixed, seed=ypea.StudioRng(9)), alg.solve(mixed, seed=ypea.StudioRng(9))
    assert np.array_equal(a.history.trace_obj_value, b.history.trace_obj_value)
    assert np.array_equal(a.best.position, b.best.position)


def tsp(n=40):
    xy = np.random.default_rng(1).random((n, 2))
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    return ypea.Problem(ypea.Permutation("t", n=n), lambda s: float(d[s.t, np.roll(s.t, -1)].sum()))


def knapsack(n=40):
    r = np.random.default_rng(2)
    w, v = r.integers(10, 60, n), r.integers(10, 60, n)
    cap = w.sum() / 3
    return ypea.Problem(ypea.Binary("b", shape=n), lambda s: (float(v @ s.b), float(w @ s.b - cap)), sense="max")


def routes(n=30, k=3):
    # k routes from a depot, modeled as a Partition (items and separators in one ordering)
    xy = np.random.default_rng(1).random((n + 1, 2))
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)

    def length(s):
        return float(sum(d[r[:-1], r[1:]].sum() for r in (np.r_[n, g, n] for g in s.q)))

    return ypea.Problem(ypea.Partition("q", n=n, k=k), length)


@pytest.mark.parametrize("name", NATIVE)
def test_native_beats_encoded(name):
    def mean(problem, space):
        alg = NATIVE[name](max_nfe=5000)
        alg.space = space
        rs = [alg.solve(problem, seed=s) for s in range(3)]
        assert all(r.best.feasible for r in rs)
        return np.mean([r.best.obj_value for r in rs])

    assert mean(tsp(), "native") < mean(tsp(), "encoded")
    assert mean(knapsack(), "native") > mean(knapsack(), "encoded")
    assert mean(routes(), "native") < mean(routes(), "encoded")


def module_swap_only(x, rng):
    return PermutationOperators(None, mutations=(swap,)).mutate(x, rng)


def test_user_operators_and_pickling():
    problem = ypea.Problem([ypea.Permutation("p", n=8), ypea.Real("x", shape=2, lb=-5, ub=5)],
                           lambda s: float(np.sum(s.p * np.arange(8))) + sphere_obj(s))
    alg = ypea.GA(space="native", max_iter=5, operators={"p": (None, module_swap_only)})
    a = alg.solve(problem, seed=4)
    b = pickle.loads(pickle.dumps(alg)).solve(problem, seed=4)
    assert np.array_equal(a.history.trace_obj_value, b.history.trace_obj_value)
    assert a.history.trace_obj_value.tolist() != ypea.GA(space="native", max_iter=5).solve(
        problem, seed=4).history.trace_obj_value.tolist()
    with pytest.raises(ValueError, match="unknown variables"):
        ypea.GA(space="native", operators={"nope": (None, swap)}, max_iter=2).solve(problem, seed=0)
    with pytest.raises(ValueError, match="space"):
        ypea.SA(space="decoded")


def test_native_mutation_changes_each_variable_with_probability_one_over_k():
    problem = ypea.Problem([ypea.Permutation(f"p{i}", n=6) for i in range(4)],
                           lambda s: float(sum(np.sum(getattr(s, f"p{i}") * np.arange(6)) for i in range(4))))
    ops = NativeOperators(problem)
    X = np.random.default_rng(0).random((4000, problem.code_size))
    Y = ops.apply("mutate", [X], np.random.default_rng(1), lambda c: (c,))[0]
    changed = np.stack([(Y[:, a:b] != X[:, a:b]).any(axis=1) for a, b in ops.slices], axis=1)
    assert changed.any(axis=1).all()  # at least one variable per mutant
    # P(changed) = 1/4 plus the forced pick when none was drawn: 1/4 + (3/4)^4 / 4 = 0.329
    assert np.allclose(changed.mean(axis=0), 0.25 + 0.75**4 / 4, atol=0.03)
    one = ypea.Problem(ypea.Permutation("p", n=6), lambda s: 0.0)
    X1 = X[:, :6]
    Y1 = NativeOperators(one).apply("mutate", [X1], np.random.default_rng(1), lambda c: (c,))[0]
    Z1 = PermutationOperators(one.vars[0]).mutate(one.vars[0].decode(X1), np.random.default_rng(1))
    assert np.array_equal(one.vars[0].decode(Y1), Z1)  # one variable: no extra draws
