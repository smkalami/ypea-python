"""Minimize the sphere function with Particle Swarm Optimization.

A Python version of doc_ypea_pso.m of YPEA for MATLAB. Run it as

    python sphere_pso.py [output folder]

It prints the best solution and saves the convergence plot (needs matplotlib).
"""

import sys
from pathlib import Path

import ypea

# 20 real variables in [-10, 10], and the sphere function as the objective
problem = ypea.Problem(
    vars=ypea.Real("x", shape=20, lb=-10, ub=10),
    objective=lambda sol: ypea.benchmarks.sphere(sol.x),
    name="sphere",
)

# the settings of the MATLAB example
pso = ypea.PSO(max_iter=100, pop_size=100, w=0.5, wdamp=1, c1=1, c2=2)
result = pso.solve(problem, seed=1)

print(result.summary())
print("best x:", result.best.solution.x.round(4))

# the constriction coefficients of Clerc and Kennedy (2002) replace w, c1 and c2
pso.use_constriction_coeffs(phi1=2.05, phi2=2.05)
print(f"constriction coefficients: w = {pso.w:.4f}, c1 = {pso.c1:.4f}, c2 = {pso.c2:.4f}")
result2 = pso.solve(problem, seed=1)
print(f"best value with constriction coefficients: {result2.best.obj_value:.4g}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/sphere_pso")
out.mkdir(parents=True, exist_ok=True)
ax = ypea.plot.convergence(result, result2, labels=["PSO", "PSO with constriction"])
ax.set_title(f"Results of {pso.full_name}")
ax.figure.savefig(out / "convergence.png", dpi=120)
plt.close(ax.figure)
print("saved", out / "convergence.png")
