# Biogeography-based Optimization (BBO)

::: ypea.algorithms.bbo.BBO

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

## Example

Minimize the sphere function in 20 dimensions with the settings of the YPEA for MATLAB example (`examples/sphere_all.py` runs every algorithm):

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.BBO(max_iter=1000, pop_size=20, keep_rate=0.5, alpha=0.9, mutation_prob=0.1,
               mutation_step_size=0.2, mutation_step_size_damp=0.98)
result = alg.solve(problem, seed=1)
print(result.summary())
```
