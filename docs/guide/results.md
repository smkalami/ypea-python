# Results and plots

## The result of a run

`solve` returns a `ypea.Result`:

```python
import json

import numpy as np

import ypea

problem = ypea.benchmarks.problem("rastrigin", 10)
result = ypea.LSHADE(max_nfe=20_000).solve(problem, seed=1)
print(result.summary())
```

```text
L-SHADE on rastrigin_d10
  best objective value: 4.45623e-07
  error: 4.45623e-07
  feasible: True (violation 0)
  evaluations: 20000, iterations: 430, time: 0.0844 s, stopped by: max_nfe
```

| Field | Meaning |
|---|---|
| `best` | the best solution (an `Individual`, see below), by the problem's comparison rules |
| `best_feasible` | the best feasible solution by objective value alone, or None if no feasible solution was found |
| `front` | with several objectives, the non-dominated solutions (see [Multiple objectives](multi-objective.md)). `best` and `best_feasible` are then None |
| `evaluations` | every evaluated solution, when the run was started with `keep_evaluations=True` (see below) |
| `history` | the progress of the run (see below) |
| `algorithm`, `params` | the short name of the algorithm and its settings |
| `problem` | the name of the problem |
| `seed` | the seed given to `solve` |
| `nfe`, `iterations` | evaluations and iterations used |
| `run_time` | seconds |
| `stop_reason` | which rule stopped the run (see [stopping rules](algorithms.md#stopping-rules)) |
| `optimum`, `error` | the known optimum of the problem and `abs(f - optimum)`, or None |
| `extra` | information of the algorithm, for example the restarts of IPOP-CMA-ES |

The best solution has these fields:

```python
best = result.best
print(best.obj_value, best.violation, best.feasible)
print(best.solution.x.round(3))  # the decoded solution
print(best.position.shape)       # its codes in [0, 1]
```

```text
4.456232796932979e-07 0.0 True
[ 0.  0.  0. -0. -0. -0.  0. -0. -0.  0.]
(10,)
```

With the default feasibility rules, `best_feasible` is the same solution as `best` once any feasible solution exists. With `constraints="penalty"` the best by penalized value may be infeasible, and `best_feasible` then gives the best solution that satisfies every constraint. Both are tracked exactly after every evaluation, and `summary()` shows the best feasible value when the best solution is infeasible.

## All evaluations

`solve(problem, keep_evaluations=True)` (and `ask_tell` with the same argument) keeps every evaluated solution in `result.evaluations`. It is off by default, because it needs memory for every evaluation. Rows past the `max_nfe` budget are not evaluated, so they are not kept.

```python
result = ypea.LSHADE(max_nfe=20_000).solve(problem, seed=1, keep_evaluations=True)
ev = result.evaluations
print(len(ev), ev.nfe[:3], ev.obj_values.shape, ev.positions.shape)
good = ev[ev.feasible & (ev.obj_values < 1)]       # filter like an array
first = good[np.argsort(good.nfe)][:1]              # sort by any field
print(first.nfe, first.obj_values.round(3))
```

```text
20000 [1 2 3] (20000,) (20000, 10)
[18476] [0.947]
```

| Field | Shape | Meaning |
|---|---|---|
| `nfe` | (n,) | the evaluation number of each row, 1, 2, ... |
| `positions` | (n, code_size) | the codes. Decode them with `problem.decode(ev.positions)` |
| `obj_values` | (n,) or (n, m) | the objective values (non-finite values are stored as the worst value) |
| `violations` | (n,) | the total constraint violations |
| `feasible` | (n,) | True where the violation is 0 |

## History

`result.history` has one entry per iteration (iteration 0 is the initialization) and one entry per improvement of the best solution:

| Field | One value per |
|---|---|
| `iteration`, `nfe`, `best_obj_value`, `best_violation`, `time` | iteration |
| `trace_nfe`, `trace_obj_value`, `trace_violation` | improvement, with the exact evaluation count that found it |
| `trace_feasible_nfe`, `trace_feasible_obj_value` | improvement of the best feasible solution |

`history.at_nfe(points)` gives the best value found within each number of evaluations, which is how experiments record values at fixed budgets:

```python
print(result.history.at_nfe([100, 1_000, 10_000, 20_000]).round(3))
```

```text
[103.416  76.667  24.616   0.   ]
```

## Saving a result

`result.to_dict()` returns plain Python data (lists, numbers and strings) ready for JSON. Values that JSON cannot hold, such as inf, become strings:

```python
with open("result.json", "w") as f:
    json.dump(result.to_dict(), f)
```

The experiment runner saves every run this way.

## Plots

`ypea.plot` needs matplotlib (`pip install "ypea[plot]"`). `ypea.plot.convergence` draws the best value against the evaluations, and takes several results at once:

```python
results = [alg.solve(problem, seed=1) for alg in (ypea.DE(max_nfe=20_000), ypea.PSO(max_nfe=20_000))]
ax = ypea.plot.convergence(*results, error=True)
ax.set_title("Rastrigin, 10 dimensions")
ax.figure.savefig("convergence.png", dpi=150)
```

| Option | Meaning |
|---|---|
| `x="nfe"` or `x="iter"` | evaluations (default) or iterations on the horizontal axis |
| `error=True` | draw the distance to the known optimum |
| `log` | a logarithmic vertical axis (default: when every value is positive) |
| `labels` | one label per result (default: the algorithm names) |
| `ax` | draw on existing matplotlib axes |

<figure markdown="span">
  ![Convergence of DE and PSO on Rastrigin](../assets/figures/convergence.svg)
  <figcaption>The plot made by the code above: the error of the best solution of DE and PSO against the evaluations.</figcaption>
</figure>

Each function returns the matplotlib `Axes`, so the figure can be changed and saved as PNG, PDF or SVG. Three more functions, `band`, `boxes` and `rank_bars`, draw the results of many runs. [Experiment reports](experiments.md) use them.

To draw a tour or the routes of a vehicle routing solution, use `ypea.problems.plot_tour` and `ypea.problems.plot_routes` (see [Test problems](test-problems.md)).
