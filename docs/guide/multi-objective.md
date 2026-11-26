# Multiple objectives

A problem may have several objectives. There is then usually no single best solution, but a set of trade offs: the Pareto front. ypea keeps that set for you while the algorithm runs.

## Defining the problem

Give the number of objectives with `n_obj`, or a list with one sense per objective. The objective returns a list or array of values (not a tuple, because a tuple means `(f, g)` or `(f, g, h)`):

```python
import numpy as np
import ypea


def zdt1_like(sol):
    x = sol.x
    g = 1 + 9 * np.mean(x[1:])
    return [x[0], g * (1 - np.sqrt(x[0] / g))]


problem = ypea.Problem(ypea.Real("x", shape=5, lb=0, ub=1), zdt1_like, n_obj=2, name="zdt1_like")
result = ypea.RandomSearch(max_nfe=5_000).solve(problem, seed=1)
print(result.summary())
```

```text
RS on zdt1_like
  front: 19 solutions, 19 feasible
  objective ranges: [1.13392e-05, 0.910699], [0.428198, 6.46732]
  evaluations: 5000, iterations: 49, time: 0.0952 s, stopped by: max_nfe
```

| Setting | Meaning |
|---|---|
| `n_obj=m` | the number of objectives. A single `sense` applies to all of them |
| `sense=["min", "max"]` | one sense per objective (`n_obj` then follows from the list) |

Constraints work as with one objective: return `(f, g)` or `(f, g, h)`, where `f` is the list of objective values. A vectorized objective returns an array of shape (n, m).

## Algorithms

Each algorithm declares the problems it solves in its `objectives` attribute: `"single"`, `"multi"` or `"both"`. The single objective algorithms raise a clear error on a problem with several objectives. `NSGAII`, `NSGAIII`, `MOEAD`, `MOPSO` and `SPEA2` solve only problems with several objectives, and `RandomSearch` solves both kinds, which makes it a baseline for multi-objective comparisons.

To use a single objective algorithm anyway, turn the problem into one objective with `ypea.scalarize`:

```python
single = ypea.scalarize(problem, weights=[0.5, 0.5])                          # weighted sum
tcheb = ypea.scalarize(problem, [0.5, 0.5], "tchebycheff", ideal=[0, 0])      # Tchebycheff
result = ypea.LSHADE(max_nfe=5_000).solve(tcheb, seed=1)
print(result.best.solution.x.round(3), zdt1_like(result.best.solution))
```

```text
[0.382 0.    0.    0.    0.   ] [np.float64(0.381966011266143), np.float64(0.38196601128882585)]
```

The weighted sum minimizes `sum(w_i * f_i)`, with the sign of each objective turned to minimization. It reaches only the convex parts of a front. Tchebycheff minimizes `max(w_i * |f_i - z_i|)` around an ideal point `z`, a point at least as good as the best value of each objective, and can reach every point of the front. Each weight vector gives one point of the front.

## The front

The run keeps an archive of every non-dominated solution found, updated exactly after every evaluation (the way the best solution is tracked with one objective). `result.front` holds it, sorted by the first objective:

```python
front = ypea.RandomSearch(max_nfe=5_000).solve(problem, seed=1).front
print(len(front), front.obj_values[:3].round(3))
print(front[0].solution.x.round(3), front[0].obj_value.round(3), front[0].feasible)
```

```text
19 [[0.    6.467]
 [0.    5.841]
 [0.    5.603]]
[0.    0.697 0.19  0.804 0.742] [0.    6.467] True
```

| Field | Shape | Meaning |
|---|---|---|
| `positions` | (k, code_size) | the codes in [0, 1] |
| `obj_values` | (k, m) | the objective values |
| `violations` | (k,) | the total constraint violations |
| `solutions` | k | the decoded solutions |
| `feasible` | (k,) | True for each feasible solution |

`front[i]` gives an `Individual` whose `obj_value` is an array. With several objectives `result.best`, `result.best_feasible` and `result.error` are None, because no single solution is best.

**Which solutions the front keeps.** The front uses constrained domination (Deb et al. 2002), whatever the problem's `constraints` setting:

1. Once a feasible solution exists, the front holds only feasible solutions, and a solution enters when no member dominates it.
2. Before that, the front holds the least violating solutions: a solution with a smaller total violation replaces the members, and of solutions with exactly the same violation the Pareto dominated ones are dropped.
3. A solution with exactly the same objective values and violation as a member does not enter, so the front has no duplicates.

Rows past the `max_nfe` budget are never evaluated and never enter the front.

**Limiting the size.** The front grows without limit by default. An algorithm that supports several objectives takes `max_archive`, for example `ypea.RandomSearch(max_archive=100)`. When the front grows larger, the member with the smallest crowding distance (the most crowded one) is removed, the distances are computed again, and so on until the front fits. The extreme solutions of each objective have an infinite distance, so they stay. The front is trimmed after each batch of evaluations.

## History and trace

`result.history` keeps one entry per iteration, with `archive_size` (the size of the front) and `best_violation` (the smallest violation in the front). `best_obj_value` is NaN.

The trace logs the front exactly: one entry per solution that entered it, with `trace_nfe` (the evaluation that found it), `trace_obj_value` (shape (entries, m)), `trace_violation`, and `trace_removed_nfe` (the evaluation at which it left the front, or -1). `history.front_at(nfe)` rebuilds the front at any budget, for example to compute an indicator over the run:

```python
h = ypea.RandomSearch(max_nfe=5_000).solve(problem, seed=1).history
for nfe in (100, 1_000, 5_000):
    print(nfe, h.front_at(nfe).sum())
```

```text
100 6
1000 8
5000 19
```

## Comparisons, sorting and crowding

`problem.comparator` compares solutions with several objectives, and algorithms call it through these helpers:

| Method | Gives |
|---|---|
| `dominates(F1, V1, F2, V2)` | element wise constrained domination (or Pareto dominance of the penalized values with `constraints="penalty"`) |
| `nondominated_sort(F, V)` | the front of each solution: 0 for the non-dominated ones, 1 for the next, and so on (fast non-dominated sorting, Deb et al. 2002) |
| `crowding_distance(F, ranks)` | the crowding distance of Deb et al. (2002) within each front: normalized by the range of each objective, infinite at the boundaries |

With `constraints="penalty"` the comparator compares the penalized values `f_i + penalty * violation` (each objective turned to minimization first) by Pareto dominance, and equal penalized values by the violation. This is the simplest rule that keeps a penalty working as with one objective. The front itself always uses constrained domination, so it holds feasible solutions only, once any exist.

## Ask and tell

`tell` takes one list or array of objective values per solution (or `(f, g)`), or an array of shape (n, m) for a vectorized problem:

```python
run = ypea.RandomSearch(max_nfe=200).ask_tell(problem, seed=1)
while not run.done:
    run.tell([zdt1_like(s) for s in run.ask()])
print(len(run.result().front))
```

```text
11
```

## Comparing algorithms

`ypea.Experiment` runs several algorithms on several problems and reports IGD and hypervolume, with the same statistical tests as for one objective. See [Experiments](experiments.md#several-objectives). `ypea.benchmarks.multi_problem` gives the ZDT, DTLZ and WFG benchmarks as problems with their true fronts.

## Operators for multi-objective algorithms

The multi-objective genetic algorithms use simulated binary crossover and polynomial mutation (Deb et al. 2002) by default. Both are in `ypea.operators` as `SBX` and `PolynomialMutation`, and work on the [0, 1] codes, so they apply to every variable type. See [Native operators](operators.md#operators-as-settings).
