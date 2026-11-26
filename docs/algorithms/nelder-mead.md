# Nelder-Mead Simplex (NM)

::: ypea.algorithms.nelder_mead.NelderMead

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

## Example

Minimize the sphere function in 20 dimensions with a budget of evaluations:

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.NelderMead(max_nfe=2_000)
result = alg.solve(problem, seed=1)
print(result.summary())
```
