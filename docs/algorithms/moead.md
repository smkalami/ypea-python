# Multiobjective Evolutionary Algorithm Based on Decomposition (MOEA/D)

::: ypea.algorithms.moead.MOEAD

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings). See [Multiple objectives](../guide/multi-objective.md) for problems with several objectives and the front.

## Example

Approximate the front of ZDT1 with 30 variables and measure its distance to the true front:

```python
import ypea
from ypea.benchmarks import pareto_front, zdt1
from ypea.indicators import igd

problem = ypea.Problem(ypea.Real("x", shape=30, lb=0, ub=1), lambda sol: zdt1(sol.x),
                       vectorized=True, n_obj=2, name="zdt1")

result = ypea.MOEAD(max_nfe=10_000).solve(problem, seed=1)
print(len(result.front), igd(result.front.obj_values, pareto_front("zdt1", 500)))
```

## Batch evaluation (a variant)

By default MOEA/D follows the paper: each child is evaluated on its own and updates the population before the next child is bred, so the objective sees one solution per call. With `batch=True`, each iteration breeds all N children from the population as it stands at the start of the iteration, evaluates them in one call, and then applies the ideal point and neighbor updates of the children in subproblem order. A vectorized objective, or one that runs the solutions in parallel, then gets N solutions at once:

```python
result = ypea.MOEAD(max_nfe=10_000, batch=True).solve(problem, seed=1)
```

This variant is not in the paper. Its children cannot use the improvements made earlier in the same iteration, so it may need a few more evaluations for the same quality. On ZDT1 and DTLZ2 with 10 000 evaluations, both modes reach a similar IGD. The default results do not change.
