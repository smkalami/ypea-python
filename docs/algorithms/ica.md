# Imperialist Competitive Algorithm (ICA)

::: ypea.algorithms.ica.ICA

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

## Example

Minimize the sphere function in 20 dimensions with the settings of the YPEA for MATLAB example (`examples/sphere_all.py` runs every algorithm):

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.ICA(max_iter=1000, pop_size=100, empire_count=10, selection_pressure=1,
               assimilation_coeff=1.5, revolution_prob=0.05, revolution_rate=0.1,
               revolution_step_size=0.1, revolution_step_size_damp=0.99, zeta=0.2)
result = alg.solve(problem, seed=1)
print(result.summary())
```
