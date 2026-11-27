# Ask and tell

`solve` calls the objective function itself. When the solutions are evaluated somewhere else (a lab experiment, a job on a cluster, a simulator that runs on its own), use the ask and tell interface instead: the algorithm asks for a batch of solutions, you evaluate them, and you tell it the values.

```python
import numpy as np
import ypea

# objective=None: the solutions are evaluated outside the toolbox
problem = ypea.Problem(ypea.Real("x", shape=5, lb=-10, ub=10), objective=None, name="outside")

run = ypea.PSO(max_nfe=500).ask_tell(problem, seed=0)
while not run.done:
    solutions = run.ask()  # decoded solutions, as the objective would get them
    values = [np.sum(s.x**2) for s in solutions]  # evaluate them anywhere
    run.tell(values)  # one value per solution, in the same order
result = run.result()
print(result.nfe, result.stop_reason)
```

```text
500 max_nfe
```

`result()` gives the same `Result` as `solve`. Told the true objective values, an ask and tell run gives exactly the result of `solve` with the same seed, for every algorithm and also with `ypea.StudioRng`.

## What ask gives and tell takes

`ask()` returns the next batch in the form the objective would get it: a list of `Solution` objects, or for a vectorized problem one `Solution` whose attributes have a leading batch axis. The codes of the batch, with shape (n, code_size), are in `run.codes`.

`tell(values)` takes what the objective would return, in the same order:

* one value `f` per solution, or `(f, g)` with constraint values `g` (satisfied when `g <= 0`), or `(f, g, h)` with equality constraints `h`, as described in [Objectives and constraints](objectives.md);
* for a vectorized problem, the arrays `f`, `(f, g)` or `(f, g, h)` of the whole batch;
* with several objectives, a list or array of values as `f` (shape (n, m) for a vectorized problem), see [Multiple objectives](multi-objective.md).

`ask_tell(problem, seed=..., keep_evaluations=True)` keeps every told value in `result().evaluations`, as `solve` does.

```python
def design(sol):
    # minimize (x - 2)^2 + (y - 1)^2 subject to x + y <= 2
    x, y = sol.x[..., 0], sol.x[..., 1]
    return (x - 2) ** 2 + (y - 1) ** 2, x + y - 2


problem = ypea.Problem(ypea.Real("x", shape=2, lb=-5, ub=5), objective=None, vectorized=True)
run = ypea.DE(max_nfe=1000, pop_size=20).ask_tell(problem, seed=1)
while not run.done:
    batch = run.ask()
    run.tell(design(batch))  # arrays for the whole batch
print(round(run.result().best.obj_value, 3))
```

```text
0.505
```

The batch size changes from call to call. Population methods usually ask for a whole population, while Nelder-Mead asks for one solution at a time after its first simplex. The run never asks for more evaluations than `max_nfe`, so the last batch can be smaller. On a problem with constraints, a run whose first batch has fewer than 20 solutions then asks for a batch of random solutions, from which it takes the scales of the constraints (see [Scaling of the constraints](objectives.md#comparing-solutions-with-constraints)).

A problem may still have an objective. Then you can evaluate some solutions with it and others elsewhere. A problem with `objective=None` works only with `ask_tell`, and `solve` raises an error for it.

## Stopping early

`run.stop()` ends the run at once and returns its result, with the stop reason `"stopped"`, or the reason you give:

```python
problem = ypea.Problem(ypea.Real("x", shape=5, lb=-10, ub=10), objective=None)
run = ypea.GA(max_iter=100, pop_size=20).ask_tell(problem, seed=0)
for _ in range(5):
    run.tell([np.sum(s.x**2) for s in run.ask()])
result = run.stop("out of lab time")
print(result.nfe, result.stop_reason)
```

```text
60 out of lab time
```

A batch that was asked for but not told is not counted. A run stopped before any value was told has no result, so `stop()` raises an error then.

`max_iter`, `max_nfe`, `max_time` and `target` end the run as in `solve`, and so do callbacks and `request_stop`. `max_time` counts the time you spend evaluating.

## Errors

* `tell()` before `ask()`, or `ask()` twice without a `tell()`, raises a `RuntimeError`.
* `tell()` with the wrong number of values raises a `ValueError`, and the run waits for the right values.
* An exception in the algorithm or in a callback is raised by `tell()`, and the run is then done.

## How it works

The algorithm runs its normal loop in a background thread, on a copy of the algorithm object. Each time it needs solutions evaluated, it hands the batch to `ask()` and waits until `tell()` hands back the values. So every algorithm supports ask and tell without changes. Callbacks run in that background thread, between `tell()` and the next batch. `run.algorithm` is the copy that runs, for example to read `run.algorithm.iter` or `run.algorithm.best` between batches. When a run is no longer used, its thread ends.
