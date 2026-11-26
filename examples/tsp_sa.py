"""Solve a traveling salesman problem (TSP) with Simulated Annealing.

A Python version of doc_ypea_sa.m of YPEA for MATLAB: 30 cities on a circle of radius 40,
in shuffled order. The shortest tour follows the circle. Run it as

    python tsp_sa.py [output folder]

It prints the best tour and saves a plot of it (needs matplotlib).
"""

import sys
from pathlib import Path

import numpy as np

import ypea

# the cities: 30 points on a circle, in a random order
n, radius = 30, 40.0
rng = np.random.default_rng(0)
angle = 2 * np.pi * rng.permutation(n) / n
x = 50 + radius * np.cos(angle)
y = 50 + radius * np.sin(angle)
d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])  # distance matrix


def tour_length(sol):
    tour = sol.tour  # a permutation of 0..n-1
    return d[tour, np.roll(tour, -1)].sum()


problem = ypea.Problem(ypea.Permutation("tour", n), tour_length, name="circle_tsp30")
optimum = 2 * n * radius * np.sin(np.pi / n)  # the perimeter of the regular polygon

# the settings of the MATLAB example. The run stops when it finds the optimal tour.
sa = ypea.SA(
    max_iter=100, pop_size=5, max_sub_iter=10, move_count=15,
    mutation_rate=1 / problem.code_size, mutation_step_size=0.5, mutation_step_size_damp=1,
    initial_temp=10_000, final_temp=1,
    target=optimum * (1 + 1e-9),  # a small tolerance for rounding in the sum of distances
)
result = sa.solve(problem, seed=1)

print(result.summary())
print("best tour:", result.best.solution.tour)
print(f"optimal length: {optimum:.4f}")

# ypea.problems.circle_tsp builds the same kind of problem in one line
same_kind = ypea.problems.circle_tsp(30, radius=40, seed=0)
print(same_kind.name, "optimum:", round(same_kind.optimum, 4))

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/tsp_sa")
out.mkdir(parents=True, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
tour = np.append(result.best.solution.tour, result.best.solution.tour[0])
ax1.plot(x[tour], y[tour], "-o", markersize=4)
ax1.set_aspect("equal")
ax1.set_title(f"Best tour, length {result.best.obj_value:.2f}")
ypea.plot.convergence(result, ax=ax2, log=False)
ax2.set_ylabel("Tour length")
fig.savefig(out / "tsp_sa.png", dpi=120)
plt.close(fig)
print("saved", out / "tsp_sa.png")
