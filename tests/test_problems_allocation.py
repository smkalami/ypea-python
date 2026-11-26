import itertools
import pickle

import numpy as np
import pytest

import ypea
from ypea.problem import Solution, parse_batch
from ypea.problems.gap import _DATA, gap, orlib_gap, read_orlib_gap
from ypea.problems.hub import cab, hub_allocation, hub_location
from ypea.problems.transportation import transportation
from ypea.problems.vrp import cvrp, cvrp_routes, plot_routes, read_cvrplib

# the exact optima for the checks come from SciPy's solvers
scipy_optimize = pytest.importorskip("scipy.optimize", reason="needs pip install ypea[stats]")
Bounds, LinearConstraint = scipy_optimize.Bounds, scipy_optimize.LinearConstraint
linprog, milp = scipy_optimize.linprog, scipy_optimize.milp


def _batch(p, X):
    return parse_batch(p.objective(p.decode(X)), len(X))


# CVRP

COORDS = np.array([[0, 0], [2, 1], [3, -1], [-1, 3], [-2, -2], [1, 4]], dtype=float)
DEMANDS = np.array([0, 4, 3, 5, 2, 4], dtype=float)


def _route_length(d, route):
    tour = [0, *route, 0]
    return sum(d[a, b] for a, b in itertools.pairwise(tour))


def test_cvrp_brute_force():
    p = cvrp(COORDS, DEMANDS, capacity=10, n_vehicles=2)
    d = p.objective.distances
    # independent optimum: split customers into two vehicles, best order of each route
    best = np.inf
    for labels in itertools.product(range(2), repeat=5):
        groups = [[c + 1 for c in range(5) if labels[c] == k] for k in range(2)]
        if any(DEMANDS[g].sum() > 10 for g in groups):
            continue
        best = min(best, sum(min(_route_length(d, r) for r in itertools.permutations(g)) for g in groups))
    perms = np.array(list(itertools.permutations(range(6))))
    F, V = parse_batch(p.objective(Solution(route=perms)), len(perms))
    assert F[V == 0].min() == pytest.approx(best)
    # the best permutation decodes to feasible routes with this length
    perm = perms[np.flatnonzero(V == 0)[F[V == 0].argmin()]]
    routes = cvrp_routes(p, perm)
    assert len(routes) == 2 and sorted(np.concatenate(routes)) == [1, 2, 3, 4, 5]
    assert sum(_route_length(d, r) for r in routes) == pytest.approx(best)
    f, v = p.evaluate(Solution(route=perm))
    assert f == pytest.approx(best) and v == 0


def test_cvrp_capacity_violation_and_pickle():
    p = cvrp(COORDS, DEMANDS, capacity=10, n_vehicles=2)
    perm = np.array([0, 1, 2, 3, 4, 5])  # all customers on vehicle 1, then the separator
    f, v = p.evaluate(Solution(route=perm))
    assert v == pytest.approx(DEMANDS.sum() - 10)
    q = pickle.loads(pickle.dumps(p))
    assert q.evaluate(Solution(route=perm)) == (f, v)


CVRP_FILE = """NAME : toy-n6-k2
COMMENT : (synthetic, Optimal value: 21)
TYPE : CVRP
DIMENSION : 6
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
 1 2 1
 2 0 0
 3 3 -1
 4 -1 3
 5 -2 -2
 6 1 4
DEMAND_SECTION
1 4
2 0
3 3
4 5
5 2
6 4
DEPOT_SECTION
 2
 -1
EOF
"""


def test_read_cvrplib(tmp_path):
    path = tmp_path / "toy.vrp"
    path.write_text(CVRP_FILE)
    p = read_cvrplib(path)
    assert p.name == "toy-n6-k2" and p.optimum == 21 and p.info["n_vehicles"] == 2
    assert np.array_equal(p.objective.demands, DEMANDS)  # depot moved to node 0
    assert np.array_equal(p.info["coords"], COORDS)
    assert np.array_equal(p.objective.distances, np.floor(np.sqrt(((COORDS[:, None] - COORDS) ** 2).sum(-1)) + 0.5))
    perms = np.array(list(itertools.permutations(range(6))))
    F, V = parse_batch(p.objective(Solution(route=perms)), len(perms))
    assert F[V == 0].min() == p.optimum
    assert read_cvrplib(path, n_vehicles=3, rounding=False).code_size == 5 + 2


def test_read_cvrplib_explicit(tmp_path):
    path = tmp_path / "x.vrp"
    path.write_text("NAME : x\nDIMENSION : 3\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : LOWER_ROW\n"
                    "CAPACITY : 5\nEDGE_WEIGHT_SECTION\n 4\n 6 3\nDEMAND_SECTION\n1 0\n2 2\n3 3\n"
                    "DEPOT_SECTION\n 1\n -1\nEOF\n")
    p = read_cvrplib(path)
    assert np.array_equal(p.objective.distances, [[0, 4, 6], [4, 0, 3], [6, 3, 0]])
    assert p.info["n_vehicles"] == 1 and p.info["coords"] is None
    assert p.evaluate(Solution(route=np.array([0, 1]))) == (13.0, 0.0)


def test_plot_routes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cvrp(COORDS, DEMANDS, capacity=10, n_vehicles=2)
    fig, ax = plt.subplots()
    assert plot_routes(p, Solution(route=np.array([0, 1, 5, 2, 3, 4])), ax=ax) is ax
    plt.close(fig)


def test_cvrp_solves():
    p = cvrp(COORDS, DEMANDS, capacity=10, n_vehicles=2)
    r = ypea.GA(max_nfe=3000).solve(p, seed=0)
    assert r.best.feasible


# GAP

def test_gap_brute_force():
    cost = np.array([[4, 7, 3, 9, 2, 6], [5, 2, 8, 4, 7, 3], [6, 5, 4, 3, 8, 1]], dtype=float)
    res = np.array([[3, 5, 2, 6, 4, 3], [4, 2, 5, 3, 6, 2], [2, 4, 3, 5, 3, 4]], dtype=float)
    cap = np.array([7, 6, 6], dtype=float)
    p = gap(cost, res, cap)
    best = np.inf
    for a in itertools.product(range(3), repeat=6):
        used = np.zeros(3)
        for j, i in enumerate(a):
            used[i] += res[i, j]
        if np.all(used <= cap):
            best = min(best, sum(cost[i, j] for j, i in enumerate(a)))
    X = np.array(list(itertools.product(range(3), repeat=6)))
    F, V = parse_batch(p.objective(Solution(machine=X)), len(X))
    assert F[V == 0].min() == best
    f, v = p.evaluate(Solution(machine=X[0]))
    assert (f, v) == (F[0], V[0])
    assert pickle.loads(pickle.dumps(p)).evaluate(Solution(machine=X[5])) == (F[5], V[5])


def test_bundled_gap_matches_milp():
    # the OR-Library optimum of c515-1, checked with an exact integer program
    problems = read_orlib_gap(_DATA / "gap_orlib_gap1.txt")
    assert [q.name for q in problems] == [f"c515-{k}" for k in range(1, 6)]
    p = orlib_gap(1)
    c, a, b = p.objective.cost, p.objective.resource, p.objective.capacity
    m, n = c.shape
    rows = np.zeros((n, m * n))
    for j in range(n):
        rows[j, j::n] = 1
    caps = np.zeros((m, m * n))
    for i in range(m):
        caps[i, i * n:(i + 1) * n] = a[i]
    out = milp(-c.ravel(), integrality=np.ones(m * n), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(rows, 1, 1), LinearConstraint(caps, -np.inf, b)])
    assert -out.fun == pytest.approx(p.optimum) == 336
    x = out.x.reshape(m, n).round().argmax(axis=0)
    f, v = p.evaluate(Solution(machine=x))
    assert f == 336 and v == 0 and p.sense == "max"


GAP_FILE = """2
 2 3
 1 2 3
 4 5 6
 1 1 1
 2 2 2
 3 4
 1 2
 9 9
 8 8
 5 5
"""


def test_read_orlib_gap(tmp_path):
    path = tmp_path / "gapx.txt"
    path.write_text(GAP_FILE)
    a, b = read_orlib_gap(path, sense="min")
    assert a.name == "c23-1" and b.name == "c12-1" and a.optimum is None
    assert np.array_equal(a.objective.cost, [[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(a.objective.capacity, [3, 4])
    assert np.array_equal(b.objective.resource, [[8, 8]])
    path.write_text(GAP_FILE[:-5])
    with pytest.raises(ValueError, match="incomplete"):
        read_orlib_gap(path)


# hub location

def _hub_cost_loop(W, C, alloc, alpha):
    n = len(W)
    return sum(W[i, j] * (C[i, alloc[i]] + alpha * C[alloc[i], alloc[j]] + C[alloc[j], j])
               for i in range(n) for j in range(n))


def test_hub_allocation_forces_hubs():
    assert np.array_equal(hub_allocation([3, 1], [1, 0, 0, 0, 0]), [1, 1, 3, 3, 3])


def test_cab_brute_force():
    p = cab(n=10, p=2, alpha=0.4)
    W, C = p.objective.flows, p.objective.costs
    assert W.sum() == pytest.approx(1.0) and C.shape == (10, 10)
    best, best_alloc = np.inf, None
    for hubs in itertools.combinations(range(10), 2):
        others = [i for i in range(10) if i not in hubs]
        alloc = np.array(list(itertools.product(range(2), repeat=8)))
        full = np.zeros((len(alloc), 10), dtype=int)
        full[:, list(hubs)] = [0, 1]
        full[:, others] = alloc
        H = np.tile(hubs, (len(alloc), 1))
        F = p.objective(Solution(hubs=H, alloc=full))
        k = F.argmin()
        if F[k] < best:
            best, best_alloc = F[k], np.array(hubs)[full[k]]
    assert _hub_cost_loop(W, C, best_alloc, 0.4) == pytest.approx(best)
    # a random check of the vectorized cost against the loop
    rng = np.random.default_rng(0)
    X = rng.random((5, p.code_size))
    F, _ = _batch(p, X)
    for i, sol in enumerate(p.decode(X).hubs):
        a = hub_allocation(sol, p.decode(X).alloc[i])
        assert F[i] == pytest.approx(_hub_cost_loop(W, C, a, 0.4))
    r = ypea.DE(max_nfe=10_000).solve(p, seed=0)
    assert r.best.obj_value <= best * 1.01


def test_hub_location_pickles():
    p = cab(n=15, p=3)
    assert p.name == "cab15_p3_a0.2" and p.code_size == 30
    X = np.random.default_rng(2).random((4, p.code_size))
    q = pickle.loads(pickle.dumps(p))
    assert np.allclose(_batch(p, X)[0], _batch(q, X)[0])
    with pytest.raises(ValueError):
        hub_location(np.ones((3, 3)), np.ones((3, 3)), p=4, alpha=0.5)


# transportation

COST = np.array([[4, 6, 9, 5], [7, 3, 5, 8], [6, 8, 3, 4]], dtype=float)
SUPPLY = np.array([30, 45, 35], dtype=float)
DEMAND = np.array([25, 30, 20, 25], dtype=float)


def _lp(cost, supply, demand):
    m, n = cost.shape
    a_eq = np.kron(np.eye(m), np.ones(n))  # row sums equal supplies
    a_ub = -np.kron(np.ones(m), np.eye(n))  # column sums at least the demands
    return linprog(cost.ravel(), A_ub=a_ub, b_ub=-demand, A_eq=a_eq, b_eq=supply, bounds=(0, None))


def test_transportation_matches_linprog():
    p = transportation(COST, SUPPLY, DEMAND)
    lp = _lp(COST, SUPPLY, DEMAND)
    assert lp.status == 0
    x = lp.x.reshape(COST.shape)
    f, v = p.evaluate(p.decode_one(p.encode({"x": x})))
    assert f == pytest.approx(lp.fun, rel=1e-9) and v <= 1e-9
    # no feasible random point beats the LP optimum
    F, V = _batch(p, np.random.default_rng(0).random((2000, p.code_size)))
    assert np.all(F[V <= 1e-9] >= lp.fun - 1e-9)
    assert np.allclose(p.decode(np.random.default_rng(1).random((3, 12))).x.sum(axis=2), SUPPLY)


def test_transportation_solves_near_lp():
    p = transportation(COST, SUPPLY, DEMAND)
    r = ypea.LSHADE(max_nfe=20_000).solve(p, seed=0)
    assert r.best.feasible and r.best.obj_value <= _lp(COST, SUPPLY, DEMAND).fun * 1.02
    assert pickle.loads(pickle.dumps(p)).evaluate(r.best.solution) == p.evaluate(r.best.solution)


def test_transportation_rejects_short_supply():
    with pytest.raises(ValueError, match="supply"):
        transportation(COST, SUPPLY / 2, DEMAND)
