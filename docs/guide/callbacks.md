# Callbacks

A callback is a function that `solve` calls after every iteration with the algorithm as its argument. Pass a list of them:

```python
import numpy as np
import ypea

problem = ypea.Problem(ypea.Real("x", shape=10, lb=-5, ub=5), lambda sol: np.sum(sol.x**2), name="sphere")

values = []
result = ypea.DE(max_iter=50).solve(problem, seed=1, callbacks=[lambda alg: values.append(alg.best.obj_value[0])])
print(len(values), values[-1] == result.best.obj_value)
```

```text
50 True
```

Inside a callback, the algorithm gives the state of the run:

| Attribute | Meaning |
|---|---|
| `alg.iter` | the current iteration |
| `alg.nfe` | evaluations so far |
| `alg.best` | the best solution so far, as a population of one (`best.obj_value[0]`, `best.V[0]` for its violation, `best.X[0]` for its codes) |
| `alg.pop` | the current population (`pop.X` codes, `pop.F` objective values, `pop.V` violations) |
| `alg.problem` | the problem, for example `alg.problem.decode_one(alg.best.X[0])` |

A callback should only read this state.

## Stopping a run

A callback that returns True stops the run, with the stop reason `"callback"`:

```python
def good_enough(alg):
    return alg.best.obj_value[0] < 1e-3


result = ypea.DE(max_iter=1000).solve(problem, seed=1, callbacks=[good_enough])
print(result.stop_reason, result.iterations)
```

```text
callback 102
```

For a fixed goal, the `target` setting does the same without a callback.

## Start and end

An object with `on_start(alg)` or `on_end(alg, result)` methods gets them called before the first iteration and after the last one. `ypea.callbacks.Print` uses them:

```python
result = ypea.DE(max_iter=30).solve(problem, seed=1, callbacks=[ypea.callbacks.Print(every=10)])
```

```text
Differential Evolution (DE) started on sphere
Iteration 10: Best Obj. Value = 8.99709, NFE = 1100
Iteration 20: Best Obj. Value = 4.28249, NFE = 2100
Iteration 30: Best Obj. Value = 1.75327, NFE = 3100
Finished: max_iter, best = 1.75327
```

A callback class with the three methods can, for example, update a plot after every iteration, as the TSP example of YPEA for MATLAB does, or write progress to a log file.
