# Harmony Search (HS)

::: ypea.algorithms.hs.HS

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

## Example

Minimize the sphere function in 20 dimensions with the settings of the YPEA for MATLAB example (`examples/sphere_all.py` runs every algorithm):

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.HS(max_iter=1000, pop_size=20, new_count=40, hmcr=0.95, par=0.1, fret_width=0.1,
              fret_width_damp=0.998)
result = alg.solve(problem, seed=1)
print(result.summary())
```
