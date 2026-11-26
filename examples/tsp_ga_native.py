"""Solve a TSP with the Genetic Algorithm on permutations (native operators).

With ``space="native"`` GA decodes each tour, applies permutation crossovers (OX, PMX and
CX) and moves (swap, inversion and insertion), and encodes the children back. The default
``space="encoded"`` works on the [0, 1] codes instead. Run it as

    python tsp_ga_native.py
"""

import ypea
from ypea.operators import PermutationOperators, inversion, order_crossover

problem = ypea.problems.random_tsp(40, seed=3)

settings = dict(max_nfe=20_000, pop_size=100, mutation_prob=0.5)
encoded = ypea.GA(**settings).solve(problem, seed=1)
native = ypea.GA(**settings, space="native").solve(problem, seed=1)

# custom operators for the variable "tour": order crossover and inversion only
var = problem.vars[0]
ops = PermutationOperators(var, crossovers=[order_crossover], mutations=[inversion])
custom = ypea.GA(**settings, space="native", operators={"tour": ops}).solve(problem, seed=1)

print(f"{problem.name}, {var.n} cities, 20 000 evaluations")
print(f"GA on the [0, 1] codes:          {encoded.best.obj_value:8.2f}")
print(f"GA with native operators:        {native.best.obj_value:8.2f}")
print(f"GA with OX and inversion only:   {custom.best.obj_value:8.2f}")

# SA can use the native moves too
sa = ypea.SA(max_nfe=20_000, initial_temp=50, final_temp=0.1, space="native")
print(f"SA with native moves:            {sa.solve(problem, seed=1).best.obj_value:8.2f}")
