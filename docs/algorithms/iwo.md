# Invasive Weed Optimization (IWO)

::: ypea.algorithms.iwo.IWO

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

## Example

Minimize the sphere function in 20 dimensions with the settings of the YPEA for MATLAB example (`examples/sphere_all.py` runs every algorithm):

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.IWO(max_iter=1000, pop_size=50, initial_pop_size=10, min_seed_count=0,
               max_seed_count=5, step_size=0.02)
result = alg.solve(problem, seed=1)
print(result.summary())
```
