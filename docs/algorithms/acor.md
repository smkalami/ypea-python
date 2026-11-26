# Ant Colony Optimization for Continuous Domains (ACOR)

::: ypea.algorithms.acor.ACOR

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

## Example

Minimize the sphere function in 20 dimensions with the settings of the YPEA for MATLAB example (`examples/sphere_all.py` runs every algorithm):

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.ACOR(max_iter=200, pop_size=10, sample_count=50, q=0.5, zeta=1)
result = alg.solve(problem, seed=1)
print(result.summary())
```
