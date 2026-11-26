"""Minimize the sphere function with every algorithm of YPEA.

Python versions of the doc_ypea_*.m scripts of YPEA for MATLAB: each algorithm uses the
settings of its MATLAB example. The algorithms that are new in YPEA for Python get a budget
of evaluations instead. Run it as

    python sphere_all.py [algorithm ...]

for example ``python sphere_all.py DE PSO``. Without names, every algorithm runs.
"""

import sys

import ypea

# vectorized=True: the objective gets the whole population at once (sol.x has shape (n, 20))
problem = ypea.Problem(
    ypea.Real("x", shape=20, lb=-10, ub=10),
    lambda sol: ypea.benchmarks.sphere(sol.x),
    vectorized=True,
    name="sphere",
)

ALGORITHMS = {
    "ABC": lambda: ypea.ABC(max_iter=100, pop_size=100, onlooker_count=20, max_acceleration=0.5),
    "ACOR": lambda: ypea.ACOR(max_iter=200, pop_size=10, sample_count=50, q=0.5, zeta=1),
    "BA": lambda: ypea.BA(max_iter=1000, pop_size=30, type="standard", selected_site_ratio=0.5,
                          selected_site_bee_ratio=0.1, elite_site_ratio=0.4, elite_site_bee_ratio=2,
                          dance_radius=0.1, dance_radius_damp=0.99),
    "BBO": lambda: ypea.BBO(max_iter=1000, pop_size=20, keep_rate=0.5, alpha=0.9, mutation_prob=0.1,
                            mutation_step_size=0.2, mutation_step_size_damp=0.98),
    "CA": lambda: ypea.CA(max_iter=100, pop_size=50, influence_type=3, accept_rate=0.2, alpha=0.9),
    "CMAES": lambda: ypea.CMAES(max_iter=100, pop_size=200),
    "DE": lambda: ypea.DE(max_iter=1000, pop_size=20, beta_min=0.1, beta_max=0.9, crossover_prob=0.1),
    "FA": lambda: ypea.FA(max_iter=1000, pop_size=10, gamma=1, beta_base=2, alpha=0.2, alpha_damp=0.99,
                          delta=0.05, m=2),
    "GA": lambda: ypea.GA(max_iter=1000, pop_size=100, crossover_prob=0.7, crossover_inflation=0.4,
                          mutation_prob=0.3, mutation_rate=0.1, mutation_step_size=0.5,
                          mutation_step_size_damp=0.99, selection="roulettewheel", selection_pressure=5),
    "HS": lambda: ypea.HS(max_iter=1000, pop_size=20, new_count=40, hmcr=0.95, par=0.1, fret_width=0.1,
                          fret_width_damp=0.998),
    "ICA": lambda: ypea.ICA(max_iter=1000, pop_size=100, empire_count=10, selection_pressure=1,
                            assimilation_coeff=1.5, revolution_prob=0.05, revolution_rate=0.1,
                            revolution_step_size=0.1, revolution_step_size_damp=0.99, zeta=0.2),
    "IWO": lambda: ypea.IWO(max_iter=1000, pop_size=50, initial_pop_size=10, min_seed_count=0,
                            max_seed_count=5, step_size=0.02),
    "PSO": lambda: ypea.PSO(max_iter=100, pop_size=100, w=0.5, wdamp=1, c1=1, c2=2),
    "SA": lambda: ypea.SA(max_iter=100, pop_size=5, max_sub_iter=10, move_count=15),
    "TLBO": lambda: ypea.TLBO(max_iter=1000, pop_size=30),
    # new in YPEA for Python
    "JADE": lambda: ypea.JADE(max_nfe=20_000),
    "SHADE": lambda: ypea.SHADE(max_nfe=20_000),
    "LSHADE": lambda: ypea.LSHADE(max_nfe=20_000),
    "IPOPCMAES": lambda: ypea.IPOPCMAES(max_nfe=20_000),
    "NelderMead": lambda: ypea.NelderMead(max_nfe=2_000),
    "RandomSearch": lambda: ypea.RandomSearch(max_nfe=20_000),
}

names = sys.argv[1:] or list(ALGORITHMS)
for name in names:
    alg = ALGORITHMS[name]()
    result = alg.solve(problem, seed=1)
    print(f"{alg.short_name:12s} best = {result.best.obj_value:10.3e}   "
          f"NFE = {result.nfe:6d}   time = {result.run_time:.2f} s")
