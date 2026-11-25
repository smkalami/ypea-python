from ypea.algorithms.bee_colony import ABC

CASES = {
    "ABC": (lambda **k: ABC(**k), 3e-7),
    "ABC-rank": (lambda **k: ABC(fitness="rank", **k), 1e-7),
}
