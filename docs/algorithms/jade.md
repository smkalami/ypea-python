# Adaptive Differential Evolution with Optional External Archive (JADE)

::: ypea.algorithms.adaptive_de.JADE

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

The rules shared by JADE, SHADE and L-SHADE are on the page [New algorithms](../new-algorithms.md#adaptive-differential-evolution-jade-shade-l-shade).

## Example

Minimize the sphere function in 20 dimensions with a budget of evaluations:

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.JADE(max_nfe=20_000)
result = alg.solve(problem, seed=1)
print(result.summary())
```

## How it searches

<figure markdown="span">
  ![JADE solutions on Rastrigin at four iterations, and its convergence](../assets/figures/algorithm-jade.png)
  <figcaption>The current solutions of JADE with default settings on the Rastrigin function in two dimensions (seed 1), after 1, 5, 20 and 50 iterations. The cross marks the global minimum. The right panel shows the error of the best solution against the evaluations.</figcaption>
</figure>
