"""Solve the binary knapsack problem of the YPEA for MATLAB documentation.

From doc_ypea_problem.m: pack items of the given values and weights, with a total weight
of at most 20, to get the largest total value. The MATLAB version subtracts a penalty in
the objective. Here the objective returns the constraint, and YPEA handles it. Run it as

    python knapsack.py
"""

import numpy as np

import ypea

v = np.array([9, 6, 8, 7, 6, 3, 6, 1, 6, 6, 7, 9, 9, 7, 6, 9, 6, 1, 2, 8])  # values
w = np.array([5, 8, 2, 5, 6, 1, 6, 4, 1, 5, 2, 2, 2, 2, 2, 1, 6, 3, 5, 7])  # weights
W = 20  # capacity


def knapsack(sol):
    # objective and constraint: the constraint holds when its value is <= 0
    return v @ sol.x, w @ sol.x - W


problem = ypea.Problem(ypea.Binary("x", shape=len(v)), knapsack, sense="max", name="knapsack")
ga = ypea.GA(max_nfe=10_000, space="native")  # bit flips and uniform crossover on x

# Deb's feasibility rules (the default): a feasible solution beats an infeasible one
result = ga.solve(problem, seed=1)
print(result.summary())
print("items packed:", np.flatnonzero(result.best.solution.x))
print("total weight:", w @ result.best.solution.x)

# a penalty method instead: f - 100 * violation, as in the MATLAB example
penalty = ypea.Problem(ypea.Binary("x", shape=len(v)), knapsack, sense="max",
                       constraints="penalty", penalty=100, name="knapsack_penalty")
print("with a penalty:", ga.solve(penalty, seed=1).best.obj_value)

# the same instance ships with YPEA, with its known optimum of 73
ready = ypea.problems.matlab_knapsack()
result = ypea.DE(max_nfe=10_000).solve(ready, seed=1)
print(f"{ready.name}: best {result.best.obj_value:g}, optimum {ready.optimum}")
