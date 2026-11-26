"""Combinatorial test problems: TSP, knapsack, QAP, n-Queens, and the instance readers.

The small TSPLIB and QAPLIB instances below are copied inline, since the libraries are not
bundled with the package. Published optimal values are reproduced through the Problem.

Sources, credited to their authors and maintainers:

* berlin52, att48, burma14, ulysses16 and gr17 (with the berlin52 and att48 optimal tours) come
  from TSPLIB, G. Reinelt, "TSPLIB: A Traveling Salesman Problem Library", ORSA Journal on
  Computing 3(4), 376-384 (1991), http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
* nug12 and its solution come from QAPLIB, R. E. Burkard, S. E. Karisch and F. Rendl, "QAPLIB:
  A Quadratic Assignment Problem Library", Journal of Global Optimization 10, 391-403 (1997),
  https://coral.ise.lehigh.edu/data-sets/qaplib/ (instance by Nugent, Vollmann and Ruml, 1968)
"""

import itertools
import pickle

import numpy as np
import pytest

import ypea
from ypea.problem import Solution
from ypea.problems import (
    circle_tsp,
    integer_knapsack,
    matlab_knapsack,
    mknap1,
    nqueens,
    plot_tour,
    qap,
    random_tsp,
    read_orlib_mknap,
    read_qaplib,
    read_qaplib_solution,
    read_tsplib,
    read_tsplib_tour,
    tsp,
)

BERLIN52 = """NAME: berlin52
TYPE: TSP
COMMENT: 52 locations in Berlin (Groetschel)
DIMENSION: 52
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 565.0 575.0
2 25.0 185.0
3 345.0 750.0
4 945.0 685.0
5 845.0 655.0
6 880.0 660.0
7 25.0 230.0
8 525.0 1000.0
9 580.0 1175.0
10 650.0 1130.0
11 1605.0 620.0
12 1220.0 580.0
13 1465.0 200.0
14 1530.0 5.0
15 845.0 680.0
16 725.0 370.0
17 145.0 665.0
18 415.0 635.0
19 510.0 875.0
20 560.0 365.0
21 300.0 465.0
22 520.0 585.0
23 480.0 415.0
24 835.0 625.0
25 975.0 580.0
26 1215.0 245.0
27 1320.0 315.0
28 1250.0 400.0
29 660.0 180.0
30 410.0 250.0
31 420.0 555.0
32 575.0 665.0
33 1150.0 1160.0
34 700.0 580.0
35 685.0 595.0
36 685.0 610.0
37 770.0 610.0
38 795.0 645.0
39 720.0 635.0
40 760.0 650.0
41 475.0 960.0
42 95.0 260.0
43 875.0 920.0
44 700.0 500.0
45 555.0 815.0
46 830.0 485.0
47 1170.0 65.0
48 830.0 610.0
49 605.0 625.0
50 595.0 360.0
51 1340.0 725.0
52 1740.0 245.0
EOF
"""
BERLIN52_TOUR = """NAME : berlin52.opt.tour
TYPE : TOUR
DIMENSION : 52
TOUR_SECTION
1
49
32
45
19
41
8
9
10
43
33
51
11
52
14
13
47
26
27
28
12
25
4
6
15
5
24
48
38
37
40
39
36
35
34
44
46
16
29
50
20
23
30
2
7
42
21
17
3
18
31
22
-1
EOF
"""
ATT48 = """NAME : att48
COMMENT : 48 capitals of the US (Padberg/Rinaldi)
TYPE : TSP
DIMENSION : 48
EDGE_WEIGHT_TYPE : ATT
NODE_COORD_SECTION
1 6734 1453
2 2233 10
3 5530 1424
4 401 841
5 3082 1644
6 7608 4458
7 7573 3716
8 7265 1268
9 6898 1885
10 1112 2049
11 5468 2606
12 5989 2873
13 4706 2674
14 4612 2035
15 6347 2683
16 6107 669
17 7611 5184
18 7462 3590
19 7732 4723
20 5900 3561
21 4483 3369
22 6101 1110
23 5199 2182
24 1633 2809
25 4307 2322
26 675 1006
27 7555 4819
28 7541 3981
29 3177 756
30 7352 4506
31 7545 2801
32 3245 3305
33 6426 3173
34 4608 1198
35 23 2216
36 7248 3779
37 7762 4595
38 7392 2244
39 3484 2829
40 6271 2135
41 4985 140
42 1916 1569
43 7280 4899
44 7509 3239
45 10 2676
46 6807 2993
47 5185 3258
48 3023 1942
EOF
"""
ATT48_TOUR = """NAME : att48.opt.tour
COMMENT : Optimum solution  for att48
TYPE : TOUR
DIMENSION : 48
TOUR_SECTION
1
8
38
31
44
18
7
28
6
37
19
27
17
43
30
36
46
33
20
47
21
32
39
48
5
42
24
10
45
35
4
26
2
29
34
41
16
22
3
23
14
25
13
11
12
15
40
9
-1
EOF
"""
BURMA14 = """NAME: burma14
TYPE: TSP
COMMENT: 14-Staedte in Burma (Zaw Win)
DIMENSION: 14
EDGE_WEIGHT_TYPE: GEO
EDGE_WEIGHT_FORMAT: FUNCTION
DISPLAY_DATA_TYPE: COORD_DISPLAY
NODE_COORD_SECTION
   1  16.47       96.10
   2  16.47       94.44
   3  20.09       92.54
   4  22.39       93.37
   5  25.23       97.24
   6  22.00       96.05
   7  20.47       97.02
   8  17.20       96.29
   9  16.30       97.38
  10  14.05       98.12
  11  16.53       97.38
  12  21.52       95.59
  13  19.41       97.13
  14  20.09       94.55
EOF
"""
ULYSSES16 = """NAME: ulysses16.tsp
TYPE: TSP
COMMENT: Odyssey of Ulysses (Groetschel/Padberg)
DIMENSION: 16
EDGE_WEIGHT_TYPE: GEO
DISPLAY_DATA_TYPE: COORD_DISPLAY
NODE_COORD_SECTION
 1 38.24 20.42
 2 39.57 26.15
 3 40.56 25.32
 4 36.26 23.12
 5 33.48 10.54
 6 37.56 12.19
 7 38.42 13.11
 8 37.52 20.44
 9 41.23 9.10
 10 41.17 13.05
 11 36.08 -5.21
 12 38.47 15.13
 13 38.15 15.35
 14 37.51 15.17
 15 35.49 14.32
 16 39.36 19.56
 EOF
"""
GR17 = """NAME: gr17
TYPE: TSP
COMMENT: 17-city problem (Groetschel)
DIMENSION: 17
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW
EDGE_WEIGHT_SECTION
 0 633 0 257 390 0 91 661 228 0 412 227
 169 383 0 150 488 112 120 267 0 80 572 196
 77 351 63 0 134 530 154 105 309 34 29 0
 259 555 372 175 338 264 232 249 0 505 289 262
 476 196 360 444 402 495 0 353 282 110 324 61
 208 292 250 352 154 0 324 638 437 240 421 329
 297 314 95 578 435 0 70 567 191 27 346 83
 47 68 189 439 287 254 0 211 466 74 182 243
 105 150 108 326 336 184 391 145 0 268 420 53
 239 199 123 207 165 383 240 140 448 202 57 0
 246 745 472 237 528 364 332 349 202 685 542 157
 289 426 483 0 121 518 142 84 297 35 29 36
 236 390 238 301 55 96 153 336 0
EOF
"""
NUG12 = """12

0 1 2 3 1 2 3 4 2 3 4 5
1 0 1 2 2 1 2 3 3 2 3 4
2 1 0 1 3 2 1 2 4 3 2 3
3 2 1 0 4 3 2 1 5 4 3 2
1 2 3 4 0 1 2 3 1 2 3 4
2 1 2 3 1 0 1 2 2 1 2 3
3 2 1 2 2 1 0 1 3 2 1 2
4 3 2 1 3 2 1 0 4 3 2 1
2 3 4 5 1 2 3 4 0 1 2 3
3 2 3 4 2 1 2 3 1 0 1 2
4 3 2 3 3 2 1 2 2 1 0 1
5 4 3 2 4 3 2 1 3 2 1 0

0  5  2  4  1  0  0  6  2  1  1  1
5  0  3  0  2  2  2  0  4  5  0  0
2  3  0  0  0  0  0  5  5  2  2  2
4  0  0  0  5  2  2 10  0  0  5  5
1  2  0  5  0 10  0  0  0  5  1  1
0  2  0  2 10  0  5  1  1  5  4  0
0  2  0  2  0  5  0 10  5  2  3  3
6  0  5 10  0  1 10  0  0  0  5  0
2  4  5  0  0  1  5  0  0  0 10 10
1  5  2  0  5  5  2  0  0  0  5  0
1  0  2  5  1  4  3  5 10  5  0  2
1  0  2  5  1  0  3  0 10  0  2  0
"""
NUG12_SLN = """ 12  578
 12  7  9  3  4  8  11  1  5  6  10  2
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def held_karp(d):
    """Exact TSP optimum by dynamic programming over subsets (city 0 is the start)."""
    m = len(d) - 1
    full = 1 << m
    dp = np.full((full, m), np.inf)
    dp[1 << np.arange(m), np.arange(m)] = d[0, 1:]
    masks = np.arange(full)
    size = np.array([x.bit_count() for x in range(full)])
    for s in range(2, m + 1):
        layer = masks[size == s]
        for j in range(m):
            sub = layer[(layer >> j) & 1 == 1]
            dp[sub, j] = (dp[sub ^ (1 << j)] + d[1:, j + 1]).min(axis=1)
    return (dp[full - 1] + d[1:, 0]).min()


def length(problem, tour):
    return float(problem.objective(Solution(tour=np.asarray(tour))))


# TSP


@pytest.mark.parametrize(("text", "tour_text", "opt"), [(BERLIN52, BERLIN52_TOUR, 7542), (ATT48, ATT48_TOUR, 10628)])
def test_tsplib_optimal_tours(tmp_path, text, tour_text, opt):
    p = read_tsplib(write(tmp_path, "a.tsp", text))
    tour = read_tsplib_tour(write(tmp_path, "a.opt.tour", tour_text))
    n = p.vars[0].n
    assert p.optimum == opt
    assert sorted(tour) == list(range(n))
    assert p.info["coords"].shape == (n, 2)
    # the encoded tour goes through decoding and evaluation unchanged
    code = p.encode({"tour": tour})
    assert np.array_equal(p.decode_one(code).tour, tour)
    assert p.evaluate(p.decode_one(code)) == (opt, 0.0)


@pytest.mark.parametrize(("text", "opt"), [(BURMA14, 3323), (ULYSSES16, 6859), (GR17, 2085)])
def test_tsplib_small_optima_by_dynamic_programming(tmp_path, text, opt):
    p = read_tsplib(write(tmp_path, "a.tsp", text))
    assert p.optimum == opt
    assert held_karp(p.objective.distances) == opt


def test_explicit_formats(tmp_path):
    rng = np.random.default_rng(1)
    n = 5
    a = np.triu(rng.integers(1, 50, (n, n)), 1)
    d = a + a.T
    iu, il = np.triu_indices(n, 1), np.tril_indices(n, -1)
    iud, ild = np.triu_indices(n), np.tril_indices(n)
    data = {
        "FULL_MATRIX": d.ravel(),
        "UPPER_ROW": d[iu], "LOWER_ROW": d[il], "UPPER_DIAG_ROW": d[iud], "LOWER_DIAG_ROW": d[ild],
        # column formats list the matrix column by column, which is the transpose read by rows
        "UPPER_COL": d.T[il], "LOWER_COL": d.T[iu], "UPPER_DIAG_COL": d.T[ild], "LOWER_DIAG_COL": d.T[iud],
    }
    for fmt, values in data.items():
        body = " ".join(map(str, values))
        text = (f"NAME : t{n}\nTYPE : TSP\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : EXPLICIT\n"
                f"EDGE_WEIGHT_FORMAT : {fmt}\nEDGE_WEIGHT_SECTION\n{body}\nEOF\n")
        p = read_tsplib(write(tmp_path, f"{fmt}.tsp", text))
        assert np.array_equal(p.objective.distances, d), fmt
        assert p.info["coords"] is None


def test_coordinate_distance_rules(tmp_path):
    # EUC_2D rounds to the nearest integer, CEIL_2D rounds up, ATT uses its pseudo Euclidean rule
    coords = "1 0 0\n2 3 4\n3 1.2 0\n"
    expected = {"EUC_2D": [5, 1, 4], "CEIL_2D": [5, 2, 5], "MAN_2D": [7, 1, 6], "MAX_2D": [4, 1, 4],
                "ATT": [2, 1, 2]}
    for kind, values in expected.items():
        text = f"NAME: c\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: {kind}\nNODE_COORD_SECTION\n{coords}EOF\n"
        d = read_tsplib(write(tmp_path, f"{kind}.tsp", text)).objective.distances
        assert [d[0, 1], d[0, 2], d[1, 2]] == values, kind
        assert np.array_equal(d, d.T)


def test_circle_tsp_optimum():
    p = circle_tsp(30, radius=40, seed=3)
    xy = p.info["coords"] - 50
    tour = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
    assert length(p, tour) == pytest.approx(p.optimum)
    assert p.optimum == pytest.approx(2 * 30 * 40 * np.sin(np.pi / 30))
    assert p.optimum < 2 * np.pi * 40
    assert length(p, np.arange(30)) > p.optimum  # the cities are shuffled


def test_tsp_from_distances_and_validation():
    d = np.array([[0, 1, 9], [9, 0, 1], [1, 9, 0]])  # asymmetric
    p = tsp(distances=d)
    assert length(p, [0, 1, 2]) == 3
    assert length(p, [0, 2, 1]) == 27
    with pytest.raises(ValueError):
        tsp()
    with pytest.raises(ValueError):
        tsp(distances=np.zeros((2, 3)))


def test_plot_tour():
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib

    matplotlib.use("Agg")
    p = random_tsp(10, seed=0)
    ax = plot_tour(p, np.arange(10))
    assert len(ax.lines) == 2
    with pytest.raises(ValueError):
        plot_tour(tsp(distances=np.ones((3, 3))), np.arange(3))
    plt.close("all")


# knapsack


def brute_force_max(problem):
    var = problem.vars[0]
    grid = itertools.product(*[range(lo, hi + 1) for lo, hi in zip(var.lb, var.ub, strict=True)])
    x = np.array(list(grid))
    f, g = problem.objective(Solution(x=x))
    return f[np.all(g <= 0, axis=1)].max()


def test_matlab_knapsack():
    p = matlab_knapsack()
    f, g = p.objective(Solution(x=p.info["x_opt"]))
    assert f == 73 and np.all(g <= 0)
    assert p.sense == "max"
    assert brute_force_max(p) == p.optimum == 73


def test_integer_knapsack_brute_force():
    p = integer_knapsack([4, 5, 7], [3, 4, 5], 17, upper=[3, 2, 2])
    assert p.vars[0].ub.tolist() == [3, 2, 2]
    # 1, 1 and 2 copies weigh 17 and are worth 23, the best of the 36 choices
    assert brute_force_max(p) == 23


def test_mknap1_bundled_optima():
    problems = mknap1()
    assert len(problems) == 7
    assert [p.optimum for p in problems] == [3800, 8706.1, 4015, 6120, 12400, 10618, 16537]
    for p in problems[:4]:
        assert brute_force_max(p) == pytest.approx(p.optimum)


def test_read_orlib_mknap(tmp_path):
    text = "2\n 3 2 5\n 2 3 4\n 1 1 1\n 2 1 3\n 2 3\n\n 2 1 0\n 1 1\n 1 1\n 1\n"
    a, b = read_orlib_mknap(write(tmp_path, "mk.txt", text))
    assert a.objective.weights.tolist() == [[1, 1, 1], [2, 1, 3]]
    assert a.optimum == 5 and brute_force_max(a) == 5
    assert b.optimum is None and brute_force_max(b) == 1


# QAP and n-Queens


def test_qaplib_nug12(tmp_path):
    p = read_qaplib(write(tmp_path, "nug12.dat", NUG12))
    cost, perm = read_qaplib_solution(write(tmp_path, "nug12.sln", NUG12_SLN))
    assert p.optimum == cost == 578
    assert p.objective(Solution(p=perm)) == 578


def test_qap_brute_force():
    rng = np.random.default_rng(2)
    flow, dist = rng.integers(0, 10, (2, 6, 6))
    p = qap(flow, dist)
    perms = np.array(list(itertools.permutations(range(6))))
    costs = p.objective(Solution(p=perms))
    best = perms[costs.argmin()]
    assert costs.min() == sum(flow[i, j] * dist[best[i], best[j]] for i in range(6) for j in range(6))


def test_nqueens():
    p = nqueens(8)
    assert p.optimum == 0
    assert p.objective(Solution(q=np.array([0, 4, 7, 5, 2, 6, 1, 3]))) == 0
    assert p.objective(Solution(q=np.arange(8))) == 28
    perms = np.array(list(itertools.permutations(range(6))))
    assert (nqueens(6).objective(Solution(q=perms)) == 0).sum() == 4  # the 4 solutions of 6 queens
    assert nqueens(3).objective(Solution(q=np.array(list(itertools.permutations(range(3)))))).min() == 1


# common properties


def all_problems():
    return [random_tsp(12, seed=1), circle_tsp(10, seed=1), matlab_knapsack(), mknap1()[0],
            integer_knapsack([1, 2, 3], [2, 2, 2], 5, upper=4), qap(*np.random.default_rng(0).random((2, 7, 7))),
            nqueens(9)]


@pytest.mark.parametrize("problem", all_problems(), ids=lambda p: p.name)
def test_vectorized_equals_row_by_row(problem):
    X = np.random.default_rng(4).random((25, problem.code_size))
    out = problem.objective(problem.decode(X))
    f, g = out if isinstance(out, tuple) else (out, np.zeros(len(X)))
    rows = np.array([problem.evaluate(problem.decode_one(x)) for x in X])
    assert np.allclose(f, rows[:, 0])
    assert np.allclose(np.maximum(np.reshape(g, (len(X), -1)), 0).sum(axis=1), rows[:, 1])


@pytest.mark.parametrize("problem", all_problems(), ids=lambda p: p.name)
def test_problems_pickle(problem):
    copy = pickle.loads(pickle.dumps(problem))
    x = np.random.default_rng(5).random(problem.code_size)
    assert problem.evaluate(problem.decode_one(x)) == copy.evaluate(copy.decode_one(x))
    assert copy.optimum == problem.optimum


@pytest.mark.parametrize("algorithm", [ypea.GA(max_iter=30, pop_size=30), ypea.SA(max_iter=30)])
def test_short_run_improves_on_random(algorithm):
    p = random_tsp(15, seed=7)
    random_mean = p.objective(p.decode(np.random.default_rng(0).random((200, p.code_size)))).mean()
    result = algorithm.solve(p, seed=1)
    assert result.best.obj_value < 0.8 * random_mean
    assert sorted(result.best.solution.tour) == list(range(15))
