# Tuning algorithm settings

Every algorithm in ypea comes with defaults from its paper or the standard literature. They are a good start, but the best settings depend on the problems and on the evaluation budget. `ypea.tuning` finds good settings for your own problems in two ways:

* `tune`: iterated racing, the method of irace. It spends most of the budget on promising settings and drops poor ones early.

* `sweep`: a grid, a list of settings, or random settings, each run on every problem as a normal experiment, with the full report.

The statistical tests need SciPy (`pip install ypea[stats]`).

## When to tune

Tune when you will solve many similar problems (for example a family of scheduling instances), when a paper compares algorithms and each one should have a fair chance, or when the defaults clearly struggle on your problems.

Do not tune on the problems you report results for. Tuning fits the settings to the tuning problems, so results on the same problems look better than they are (see "Reporting tuned settings" below).

## How racing works

A race compares several settings on a stream of instances. An instance is one problem with one seed, and every setting gets the same seed on it, so they are compared under the same conditions.

1. All settings run on the first instances. The results on each instance are ranked: 1 for the smallest error (or the best objective value when the problem has no known optimum), and infeasible results after all feasible ones.

2. After `first_test` instances (5 by default), a Friedman test checks whether the ranks differ more than chance would explain. If they do, each setting is compared with the best one (a post hoc test on the average ranks, with the Holm correction for many comparisons). Settings that are clearly worse are dropped.

3. The race goes on, one instance at a time, until only a few settings are left or the race has used its share of the budget.

Iterated racing repeats this. The first race starts from settings drawn at random from the parameter space. The survivors (the elites) go on to the next race with all their results. New settings are drawn near the elites: a better elite is chosen as a parent more often, numbers come from a normal distribution around the parent's value that gets narrower each iteration, and categories lean more and more toward the parent's choice. Elites cannot be dropped before the new settings have seen the same instances. The tuning ends when the budget (the total number of runs) is used.

## Example

```python
import ypea
from ypea.benchmarks import problem
from ypea.tuning import tune

# tune on shifted and rotated classic functions in 10 dimensions
train = [problem(name, 10, shift=True, rotate=True, seed=1)
         for name in ("rastrigin", "ellipsoid", "rosenbrock", "ackley")]

result = tune(ypea.DE, train, budget=2000, max_nfe_per_run=20_000, n_jobs=4,
              seed=0, output="results/tune-de")
print(result.best)        # for example {"crossover_prob": 0.87, "pop_size": 42, ...}
print(result.history[0])  # settings, instances, runs and survivors of the first race

de = result.algorithm(max_nfe=20_000)   # the tuned algorithm, ready to use
```

`budget` is the number of runs, not of evaluations. Each run uses at most `max_nfe_per_run` evaluations. With `output`, every run is saved, and the same call again resumes an interrupted tuning. The result depends only on `seed`, not on `n_jobs`.

### Choosing what to tune

`ypea.tuning.space(ypea.DE)` shows the built in space: the type, the typical range and the valid range of each setting, and the settings left out on purpose with the reason (`space(ypea.DE).fixed`). Fix settings with `fixed`, or pass your own space:

```python
from ypea.tuning import Space, categorical, integer, real

space = Space({
    "crossover_prob": real(0.5, 1.0),
    "pop_size": integer(20, 200, log=True),
    "base_vector": categorical("rand", "best"),
})
result = tune(ypea.DE, train, budget=1000, space=space, fixed={"beta_min": 0.3, "beta_max": 0.7})
```

Fewer tuned settings need a smaller budget: the number of races and the number of survivors grow with the number of tuned settings. When the budget is too small for the number of tuned settings, the tuning stops early with a warning, and `result.runs` shows how many runs it used.

## Sweeps

A sweep runs every setting on every problem `runs` times and returns an experiment `Report`, with tables, statistical tests and figures:

```python
from ypea.tuning import sweep

report = sweep(ypea.DE, {"crossover_prob": [0.1, 0.5, 0.9], "pop_size": [20, 50]}, train,
               runs=10, max_nfe=20_000, n_jobs=4)
report.table().to_markdown("sweep.md")

# 20 random settings from the built in space
report = sweep(ypea.DE, None, train, n=20, runs=5, max_nfe=20_000)
```

A sweep is easier to explain than a race and shows how each setting behaves, but it needs many more runs for the same coverage.

## Reporting tuned settings fairly

1. Tune and test on different instances. For the classic functions, use other shift and rotation seeds (for example `seed=1` for tuning and `seed=2` for testing), other dimensions, or other functions. For CEC suites, a common choice is to tune on one suite and test on another.

2. Give every algorithm in a comparison the same tuning budget and the same `max_nfe_per_run`, and tune them all, or none.

3. Report the tuning setup: the space (types and ranges), the budget, `max_nfe_per_run`, the tuning instances, the seed, and the ypea version. The `output` folder holds `tune.json` with the result and the history of every iteration.

4. Report the tuned values next to the defaults, and the test results of both, so readers can see what the tuning gained.

## References

M. Lopez-Ibanez, J. Dubois-Lacoste, L. Perez Caceres, M. Birattari and T. Stutzle (2016). The irace package: Iterated racing for automatic algorithm configuration. Operations Research Perspectives 3, 43 to 58.

M. Birattari, T. Stutzle, L. Paquete and K. Varrentrapp (2002). A racing algorithm for configuring metaheuristics. Proceedings of GECCO 2002, 11 to 18.
