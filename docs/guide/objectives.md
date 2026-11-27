# Objectives and constraints

## The objective function

The objective gets one decoded `Solution` and returns one of:

- a number `f`,
- a tuple `(f, g)`, where `g` is one constraint value or a list or array of them,
- a tuple `(f, g, h)`, where `h` holds equality constraints (see [Equality constraints](#equality-constraints)).

A constraint is satisfied when its value is at most 0. Write `a <= b` as `g = a - b` and `a >= b` as `g = b - a`.

With several objectives, `f` is a list or array of values (see [Multiple objectives](multi-objective.md)).

```python
import numpy as np
import ypea

v = np.array([9, 6, 8, 7, 6, 3, 6, 1, 6, 6, 7, 9, 9, 7, 6, 9, 6, 1, 2, 8])  # values
w = np.array([5, 8, 2, 5, 6, 1, 6, 4, 1, 5, 2, 2, 2, 2, 2, 1, 6, 3, 5, 7])  # weights


def knapsack(sol):
    return v @ sol.x, w @ sol.x - 20  # total value, and total weight at most 20


problem = ypea.Problem(ypea.Binary("x", shape=20), knapsack, sense="max")
result = ypea.DE(max_nfe=10_000).solve(problem, seed=1)
print(result.best.obj_value, result.best.feasible, w @ result.best.solution.x)
```

```text
73.0 True 20
```

The total violation of a solution is the sum of the positive constraint values, `sum(max(g, 0))`, plus the violations of the equality constraints. `result.best.violation` holds it, and `result.best.feasible` is True when it is 0.

## Equality constraints

Return `(f, g, h)` for equality constraints `h(x) = 0`. `g` holds the inequalities as before, and may be None or an empty list when there are none:

```python
def circle(sol):
    x, y = sol.x
    return x + y, None, x**2 + y**2 - 1  # minimize x + y on the unit circle


problem = ypea.Problem(ypea.Real("x", shape=2, lb=-2, ub=2), circle)
result = ypea.CMAES(max_nfe=20_000).solve(problem, seed=0)
print(result.best.obj_value, result.best.feasible)
```

```text
-1.4142849783487175 True
```

**Why a tolerance is needed.** A search over real numbers almost never lands exactly on the surface `h(x) = 0`, so with an exact test no solution would ever be feasible. An equality therefore counts as satisfied when it is close to 0, as in `numpy.isclose`:

```text
|h| <= atol + rtol * scale
```

An equality that is not satisfied adds `|h| - (atol + rtol * scale)` to the total violation, like an inequality. Set the tolerances on the problem:

| Setting | Default | Meaning |
|---|---|---|
| `equality_atol` | 1e-4 | absolute tolerance, the value the CEC competitions use |
| `equality_rtol` | 1e-6 | relative tolerance, multiplied by the scale |
| `equality_scale` | 1 | the typical size of each equality, one number for all or a list with one per equality |

The minimum above is -1.41421..., and the tolerance lets the best solution go a little past the circle, so the reported value is slightly below it. A smaller `equality_atol` gives a more exact answer, but leaves a thinner feasible region for the algorithm to find.

**How to pick the scale.** The absolute tolerance suits equalities whose terms are about 1 in size. When an equality balances large quantities, for example a mass balance in kilograms around 10 000, a difference of 1e-4 is far below the precision of the model. Give the typical size of its terms as its scale, so the relative part of the tolerance takes over:

```python
def plant(sol):
    flow_in, flow_out, loss = sol.x
    cost = 3 * flow_in + 0.01 * loss
    return cost, None, [flow_in - flow_out - loss, loss - 0.02 * flow_in]


problem = ypea.Problem(ypea.Real("x", shape=3, lb=0, ub=20_000), plant,
                       equality_scale=[10_000, 100], equality_rtol=1e-5)
```

Here the first equality is satisfied within 1e-4 + 1e-5 * 10 000 = 0.1 kg, and the second within about 0.001. You can also divide each `h` by its scale yourself and keep the defaults.

## Comparing solutions with constraints

The problem compares solutions, so every algorithm handles constraints the same way, for minimization and maximization alike.

**Feasibility rules** (`constraints="feasibility"`, the default), after Deb (2000):

1. A feasible solution beats an infeasible one.
2. Of two infeasible solutions, the one with the smaller total violation wins.
3. Of two feasible solutions, the one with the better objective value wins.

The rules need no tuning. Two infeasible solutions with exactly the same violation are ordered by their objective values, a tie that Deb (2000) leaves open.

**Scaling of the constraints.** Constraints often differ in size: a stress limit in megapascals and a length in meters. Added up raw, the largest one would decide every comparison. So, as Deb (2000) recommends, each constraint is scaled before the sum. By default (`constraint_scaling=True`) a run divides each constraint by its largest violation among its first solutions, and keeps these scales for the whole run, so comparisons stay consistent. Equalities are scaled the same way, after the inequalities.

- The scales come from at least 20 solutions. When an algorithm starts with fewer (CMA-ES starts with one point, Nelder-Mead with d + 1, SA and HS with 10), the run first evaluates random solutions to make up 20. They count as evaluations, and they can become the best solution.
- A constraint that no early solution violates keeps the scale 1.
- `Result.constraint_scale` holds the scales. Every violation in the result (the best solution, the traces, the front and the evaluation log) is the scaled sum, and `problem.evaluate(solution)` still gives the raw sum.
- Feasibility does not depend on the scales: a solution is feasible exactly when every constraint is satisfied.

`constraint_scaling=False` adds the raw violations, and a list gives your own scales, one per constraint (the inequalities, then the equalities):

```python
def beam(sol):
    width, height = sol.x
    stress = 6000 / (width * height**2)  # MPa, at most 200
    weight = width * height
    return weight, [stress - 200, height / width - 4]


problem = ypea.Problem(ypea.Real("x", shape=2, lb=0.5, ub=10), beam, constraint_scaling=[200, 1])
```

**Penalty** (`constraints="penalty"`): solutions are compared by `f + penalty * violation` (in "smaller is better" terms, so for maximization the penalty lowers the value), and ties by the violation. `penalty` defaults to 1e6. The violation is the scaled one (see above), so one penalty suits constraints of any size.

```python
penalized = ypea.Problem(ypea.Binary("x", shape=20), knapsack, sense="max",
                         constraints="penalty", penalty=100)
print(ypea.DE(max_nfe=10_000).solve(penalized, seed=1).best.obj_value)
```

```text
73.0
```

`result.best.obj_value` is always the raw objective value `f`, without the penalty.

Some algorithms use objective values directly, for example the fitness of ABC and the seed counts of IWO. They follow their papers on problems without constraints, and switch to ranks from the comparison rules when the objective returns constraint values. GA, ICA and the probabilistic mode of BA always use ranks.

## Values that are not finite

An objective value that is NaN, inf or -inf counts as the worst possible value, and a NaN constraint value counts as an infinite violation. So an objective may return `np.inf` for a solution it cannot evaluate, and a run never ends with a NaN as its best value.

## Vectorized objectives

With `vectorized=True` the objective is called once per batch of solutions. Each attribute of the solution then has a leading population axis, and the objective returns an array `f` of shape (n,) (or (n, number of objectives)), `g` of shape (n,) or (n, number of constraints), and optionally `h` of shape (n,) or (n, number of equalities):

```python
def knapsack_batch(sol):
    x = sol.x  # shape (n, 20)
    return x @ v, x @ w - 20


problem = ypea.Problem(ypea.Binary("x", shape=20), knapsack_batch, sense="max", vectorized=True)
print(ypea.DE(max_nfe=10_000).solve(problem, seed=1).best.obj_value)
```

```text
73.0
```

A vectorized objective gives the same results as the plain one, often many times faster. Every function of `ypea.benchmarks` works on batches, and every ready made problem in `ypea.benchmarks` and `ypea.problems` is vectorized. Problems with a `Partition` variable cannot be vectorized, because its groups have different sizes.

## Parallel evaluation

For an expensive objective, pass a `concurrent.futures` executor to `solve`. The solutions of each batch are then evaluated in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

problem = ypea.Problem(ypea.Binary("x", shape=20), knapsack, sense="max")
with ThreadPoolExecutor(max_workers=4) as pool:
    result = ypea.DE(max_nfe=10_000).solve(problem, seed=1, executor=pool)
print(result.best.obj_value)
```

```text
73.0
```

The result is the same as without the executor. Threads help when the objective releases the GIL (NumPy, external programs, input and output). For pure Python objectives use a `ProcessPoolExecutor`. The objective must then be picklable: a function defined at the top level of a module, not a lambda, and on Windows and macOS the script needs an `if __name__ == "__main__":` guard. The executor is not used for vectorized problems.
