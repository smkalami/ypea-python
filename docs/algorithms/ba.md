# Bees Algorithm (BA)

::: ypea.algorithms.ba.BA

The common settings (`max_iter`, `pop_size`, `max_nfe`, `target`, `max_time` and `display`) are described in [Algorithms](../guide/algorithms.md#common-settings).

## Example

Minimize the sphere function in 20 dimensions with the settings of the YPEA for MATLAB example (`examples/sphere_all.py` runs every algorithm):

```python
import ypea

problem = ypea.Problem(ypea.Real("x", shape=20, lb=-10, ub=10),
                       lambda sol: ypea.benchmarks.sphere(sol.x), vectorized=True, name="sphere")

alg = ypea.BA(max_iter=1000, pop_size=30, type="standard", selected_site_ratio=0.5,
              selected_site_bee_ratio=0.1, elite_site_ratio=0.4, elite_site_bee_ratio=2,
              dance_radius=0.1, dance_radius_damp=0.99)
result = alg.solve(problem, seed=1)
print(result.summary())
```

## How it searches

<figure markdown="span">
  ![BA solutions on Rastrigin at four iterations, and its convergence](../assets/figures/algorithm-ba.png)
  <figcaption>The current solutions of BA with default settings on the Rastrigin function in two dimensions (seed 1), after 1, 5, 20 and 50 iterations. The cross marks the global minimum. The right panel shows the error of the best solution against the evaluations.</figcaption>
</figure>
