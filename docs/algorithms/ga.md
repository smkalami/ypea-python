# Genetic Algorithm (GA)

::: ypea.algorithms.ga.GA

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

With `space="native"` the algorithm uses operators made for each variable type. See [Native operators](../guide/operators.md).

## Example

Minimize the sphere function in 20 dimensions with the settings of the YPEA for MATLAB example (`examples/sphere_all.py` runs every algorithm):

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.GA(max_iter=1000, pop_size=100, crossover_prob=0.7, crossover_inflation=0.4,
              mutation_prob=0.3, mutation_rate=0.1, mutation_step_size=0.5,
              mutation_step_size_damp=0.99, selection="roulettewheel", selection_pressure=5)
result = alg.solve(problem, seed=1)
print(result.summary())
```
