# Native operators for GA and SA

By default every algorithm works on the [0, 1] codes of the variables. This works for every variable type, but for some types an operator made for the type works better: a crossover of two tours should give a tour that keeps parts of both parents.

GA and SA have a setting `space`. With `space="native"` they decode each variable, apply its native operators, and encode the children back:

```python
import ypea

problem = ypea.problems.random_tsp(40, seed=3)
for space in ("encoded", "native"):
    result = ypea.GA(max_nfe=20_000, mutation_prob=0.5, space=space).solve(problem, seed=1)
    print(f"{space:8s} {result.best.obj_value:.2f}")
```

```text
encoded  741.35
native   545.90
```

## Default operators

| Variable type | Crossover (GA) | Mutation (GA and SA) |
|---|---|---|
| `Real` | blend crossover (BLX alpha), with alpha = `crossover_inflation` | Gaussian steps of `mutation_step_size * (ub - lb)` on a share `mutation_rate` of the elements, at least one |
| `Integer` | uniform crossover | each changed element either creeps by a few steps or is reset to another value |
| `Binary` | uniform crossover | bit flips |
| `Permutation` | order crossover (OX), partially mapped crossover (PMX) or cycle crossover (CX), one picked at random for each pair | swap, inversion or insertion, one picked at random for each move |
| `Selection` | takes items of the other parent where they are not used yet | replaces an item with an unused one, or swaps two positions |
| `BinarySelection` | keeps the common ones and shares the others between the children | swaps a one and a zero |
| `Partition`, `BinaryPartition` | the `Permutation` crossovers, on the ordering of the n items and k - 1 separators | the `Permutation` moves, on the same ordering |
| `FixedSum` | the code operators of the algorithm, on the codes of that variable | the same |

Every native operator keeps the rules of its type: a permutation stays a permutation, a mask keeps exactly k ones, a partition keeps every item in exactly one group, and values stay inside their bounds.

For a partition, the ordering is rebuilt from the groups as the variable's `encode` builds it: the items of group 0, then separator n, the items of group 1, separator n + 1, and so on. So the separators are numbered by group position, and order crossover and PMX treat them as distinct values. This lets a routing problem with k vehicles be modeled as `Partition("routes", n=customers, k=vehicles)` and still use the permutation operators. In a problem with several variables, each variable gets its own operators.

With GA on a combinatorial problem, also set `mutation_step_size_damp=1`, so the step does not shrink until the search stops moving. See [GA](../algorithms/ga.md#long-runs-and-combinatorial-problems).

## Your own operators

`operators` maps a variable name to an object with `crossover(x1, x2, rng)` and `mutate(x, rng)` methods, or to a tuple `(crossover, mutate)`. The functions get decoded batches of shape (n, *shape): a crossover returns two child batches and a mutation one batch. A missing method or a None entry keeps the default.

The operator classes of `ypea.operators` take lists of the permutation operators, so a subset is one line:

```python
from ypea.operators import PermutationOperators, inversion, order_crossover

ops = PermutationOperators(problem.vars[0], crossovers=[order_crossover], mutations=[inversion])
result = ypea.GA(max_nfe=20_000, mutation_prob=0.5, space="native", operators={"tour": ops}).solve(problem, seed=1)
print(f"{result.best.obj_value:.2f}")
```

```text
531.09
```

A mutation written from scratch, for SA:

```python
import numpy as np


def reverse_segment(x, rng):
    """Reverse a random segment of each tour in the batch x."""
    y = x.copy()
    for row in y:
        i, j = np.sort(rng.integers(len(row), size=2))
        row[i:j + 1] = row[i:j + 1][::-1]
    return y


sa = ypea.SA(max_nfe=20_000, initial_temp=50, final_temp=0.1, space="native",
             operators={"tour": (None, reverse_segment)})
print(f"{sa.solve(problem, seed=1).best.obj_value:.2f}")
```

```text
524.73
```

Draw random numbers only from the `rng` argument, so the runs stay reproducible. To keep runs compatible with `ypea.StudioRng`, use only its methods: `random`, `uniform`, `integers`, `standard_normal`, `standard_cauchy`, `geometric`, `choice` and `permutation`.

## Operator functions

`ypea.operators` also has the single operators, for use in your own code:

- crossovers of arrays: `uniform_crossover(x1, x2, rng)` and `one_point_crossover(x1, x2, rng)`,
- crossovers of two permutations with cut points `i <= j`: `order_crossover`, `pmx` and `cycle_crossover`,
- moves of one permutation at positions `i` and `j`: `swap`, `inversion`, `insertion` and `scramble`.

`examples/tsp_ga_native.py` compares these choices on a TSP.
