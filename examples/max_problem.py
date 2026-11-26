"""A maximization problem with three variables, from doc_ypea_problem.m of YPEA for MATLAB.

Find x in [-2, 5], y in [-1, 1] and z in [3, 10] that maximize

    f(x, y, z) = x y - z^3 - 2 x - 5 y + 3 (x + y) z^2

Run it as ``python max_problem.py``.
"""

import ypea


def f(sol):
    x, y, z = sol.x, sol.y, sol.z
    return x * y - z**3 - 2 * x - 5 * y + 3 * (x + y) * z**2


problem = ypea.Problem(
    vars=[
        ypea.Real("x", lb=-2, ub=5),
        ypea.Real("y", lb=-1, ub=1),
        ypea.Real("z", lb=3, ub=10),
    ],
    objective=f,
    sense="max",
    name="max_problem",
)

for alg in (ypea.DE(max_iter=200, pop_size=20), ypea.PSO(max_iter=200, pop_size=20), ypea.NelderMead(max_nfe=500)):
    result = alg.solve(problem, seed=1)
    s = result.best.solution
    print(f"{alg.short_name:4s} f = {result.best.obj_value:.6f} at x = {s.x:.4f}, y = {s.y:.4f}, z = {s.z:.4f}")
