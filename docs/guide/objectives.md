# Objectives and constraints

## The objective function

The objective gets one decoded `Solution` and returns one of:

- a number `f`,
- a tuple `(f, g)`, where `g` is one constraint value or a list or array of them.

A constraint is satisfied when its value is at most 0. Write `a <= b` as `g = a - b`, `a >= b` as `g = b - a`, and an equality `a == b` as `g = abs(a - b) - tol` with a small tolerance.

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

The total violation of a solution is the sum of the positive constraint values, `sum(max(g, 0))`. `result.best.violation` holds it, and `result.best.feasible` is True when it is 0.

## Comparing solutions with constraints

The problem compares solutions, so every algorithm handles constraints the same way, for minimization and maximization alike.

**Feasibility rules** (`constraints="feasibility"`, the default), after Deb (2000):

1. A feasible solution beats an infeasible one.
2. Of two infeasible solutions, the one with the smaller total violation wins.
3. Of two feasible solutions, the one with the better objective value wins.

The rules need no tuning, and the objective value of an infeasible solution never matters.

**Penalty** (`constraints="penalty"`): solutions are compared by `f + penalty * violation` (in "smaller is better" terms, so for maximization the penalty lowers the value), and ties by the violation. `penalty` defaults to 1e6.

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

With `vectorized=True` the objective is called once per batch of solutions. Each attribute of the solution then has a leading population axis, and the objective returns an array `f` of shape (n,), and `g` of shape (n,) or (n, number of constraints):

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
