# Changelog

## Unreleased

- **Docs:** figures and diagrams throughout: how the library fits together, how each variable type decodes its codes, the run loop, the constraint rules, ask and tell, the tuning race, benchmark functions, test problem solutions, Pareto fronts, indicators, operators, experiment reports, and a figure of how each algorithm searches. `docs/figures/make_figures.py` makes every figure from seeded runs.

## 0.7.0 (2026-11-27)

- **MOPSO inertia (seeded MOPSO results change once):** the default `w` is now 0.7 (before 0.4, the paper's value). On 21 ZDT, DTLZ and WFG problems (15 runs, 25 000 evaluations), 0.7 had 13 wins, 2 ties and 6 losses in IGD and 12 wins, 4 ties and 5 losses in hypervolume. `MOPSO(w=0.4)` gives the paper's setting.
- **Constraint scaling (seeded results on constrained problems change once):** each constraint is divided by its largest violation among the first solutions of a run (at least 20, so algorithms that start with fewer evaluate random solutions to make up 20), as Deb (2000) recommends, so constraints of different sizes count alike. The scales stay fixed for the run and are in `Result.constraint_scale`. The penalty uses the scaled violation too. `Problem(constraint_scaling=False)` gives the raw sum as before, and a list gives fixed scales. See [Scaling of the constraints](guide/objectives.md#comparing-solutions-with-constraints).
- **ICA defaults from the paper (seeded ICA results change once):** `assimilation_coeff=2`, `zeta=0.1`, `empire_count=10` (before 1.5, 0.2 and 5), and the competition rule of Atashpaz-Gargari and Lucas (2007), the largest P - U(0, 1). On 119 benchmark problems (15 runs each), they had 30 wins, 83 ties and 6 losses against the old values. `competition="roulette"` gives the old draw. See [Differences from YPEA for MATLAB](differences-from-matlab.md#default-values).
- **HS and BA:** the paper values tested (one new harmony per iteration, and the Bees Algorithm's fewer sites and smaller patch) did not do better, so the defaults stay. The evidence is in [Differences from YPEA for MATLAB](differences-from-matlab.md#default-values).
- **Docs:** claims about the papers of BA, HS, ICA, TLBO, CA and FA corrected after checking them against the papers.

## 0.6.0 (2026-11-26)

- **New defaults for DE and GA (seeded results change once):** DE uses the fixed scale factor F = 0.5 of Storn and Price (`beta_min=0.5, beta_max=0.5`, before a new F per component from [0.2, 0.9]), and GA uses BLX-0.5 (`crossover_inflation=0.5`, before 0.1). On 119 benchmark problems (15 runs each), F = 0.5 had 38 wins, 81 ties and no loss, and BLX-0.5 had 41 wins, 70 ties and 8 losses. The old values stay available as settings. See [Differences from YPEA for MATLAB](differences-from-matlab.md#default-values).
- **Multiple objectives (core):** `Problem(..., n_obj=m, sense=[...])`, constrained domination, fast non-dominated sorting and the crowding distance of Deb et al. (2002) in the comparator, an exact archive of the non-dominated solutions in every run (`Result.front`, with `max_archive`), a trace that rebuilds the front at any budget (`history.front_at`), and `ypea.scalarize` (weighted sum or Tchebycheff). `RandomSearch` solves both kinds of problems, and the other algorithms raise a clear error on several objectives. See [Multiple objectives](guide/multi-objective.md). Single objective results are unchanged.
- **Operator library:** `ypea.operators.SBX` and `PolynomialMutation` (Deb's bounded forms, in the [0, 1] codes), and `NativeCrossover` and `NativeMutation`. GA takes `crossover=` and `mutation=`, and SA takes `mutation=`, with a check that each operator fits the space and the variables. The defaults and their results are unchanged. GA's method `crossover` is now `blend_crossover`.
- **Equality constraints:** the objective may return `(f, g, h)`. An equality counts as satisfied when `|h| <= atol + rtol * scale` (defaults 1e-4, 1e-6 and 1, set with `equality_atol`, `equality_rtol` and `equality_scale`). See [Equality constraints](guide/objectives.md#equality-constraints).
- **Best feasible solution and the evaluation log:** `Result.best_feasible` is the best feasible solution, tracked exactly with its own trace, also when `constraints="penalty"` makes the best solution infeasible. `solve(..., keep_evaluations=True)` keeps every evaluated solution in `Result.evaluations`.
- **Native mutation with several variables:** each variable now mutates with probability 1/k (k variables), with at least one per mutant, instead of all at once. A good move in one variable is no longer always paired with forced changes in the others. On a problem with three routes and a knapsack (10 runs, 10 000 evaluations), the mean result improved from -9.48 to -11.65 with GA and from -6.76 to -11.85 with SA. Results with one variable, or with the default `space`, are unchanged.
- **Ask and tell:** `algorithm.ask_tell(problem, seed=...)` runs any algorithm while you evaluate its solutions yourself (`ask()`, `tell(values)`, `result()`, `stop()`), with the same results as `solve`. A problem may have `objective=None` for this. See [Ask and tell](guide/ask-tell.md).
- **Multi-objective benchmarks:** ZDT1 to ZDT6, DTLZ1 to DTLZ7 and WFG1 to WFG9 as vectorized functions with sampled true Pareto fronts (`ypea.benchmarks.pareto_front`), checked against pymoo. The indicators GD, IGD, GD+, IGD+, hypervolume (exact for up to three objectives, seeded Monte Carlo above) and spacing in `ypea.indicators`, and front plots in `ypea.plot.front`. See [Multi-objective benchmarks](guide/multi-objective-problems.md).
- **NSGA-II and NSGA-III:** `ypea.NSGAII` (Deb et al. 2002) and `ypea.NSGAIII` (Deb and Jain 2014) for problems with several objectives, with the papers' operators by default and any code operator from `ypea.operators` as `crossover=` and `mutation=`. See [NSGA-II](algorithms/nsga2.md) and [NSGA-III](algorithms/nsga3.md).
- **MOEA/D** (Zhang and Li 2007): decomposition with Tchebycheff (default), weighted sum or PBI subproblems on Das and Dennis weights, SBX and polynomial mutation by default. See [MOEA/D](algorithms/moead.md).
- **MOPSO** (Coello Coello, Pulido and Lechuga 2004): a particle swarm with a repository of leaders in an adaptive grid and the paper's fading mutation. See [MOPSO](algorithms/mopso.md).
- **Experiments with several objectives:** each run stores the IGD and hypervolume of its front at the end and at the checkpoints, and `Report.table(metric="igd")` and `metric="hypervolume"` compare them with the same Wilcoxon and Friedman tests as for one objective, with convergence plots of the checkpoints. The hypervolume is normalized by the ideal and nadir points of the true front, with the reference point 1.1. `ypea.benchmarks.multi_problem` gives ZDT, DTLZ and WFG benchmarks as problems with their true fronts. Single objective runs and their stored files are unchanged. See [Experiments](guide/experiments.md#several-objectives).
- **MOEA/D `batch=True`:** a variant that breeds and evaluates a whole generation in one call, then applies the updates in order, for vectorized or parallel objectives. The default (`batch=False`, the paper) is unchanged.
- **SPEA2** (Zitzler, Laumanns and Thiele 2001): strength and density fitness, an archive with the truncation operator, SBX and polynomial mutation by default. See [SPEA2](algorithms/spea2.md).

## 0.5.0 (2026-11-26)

- The GA guide recommends `mutation_step_size_damp=1` for combinatorial problems and long runs, with benchmark evidence. The default stays 0.99.
- Native operators for `Partition` and `BinaryPartition` (`space="native"`): the permutation crossovers and moves on the ordering of items and separators, so routing problems modeled as a partition benefit from the native space. Results with the default `space` are unchanged. Native space runs with a partition variable change.

## 0.4.0 (2026-11-26)

- **Benchmarks:** 26 classic functions with seeded shift and rotation, and the CEC 2017 and CEC 2022 suites with the official data, checked against the official C code. See [Benchmarks](guide/benchmarks.md).
- **Test problems:** constrained engineering design (pressure vessel, welded beam, spring, speed reducer, three bar truss, gear train), combinatorial problems (TSP, knapsack, QAP, n-Queens) and allocation problems (CVRP, generalized assignment, hub location, transportation), with readers for TSPLIB, QAPLIB, OR-Library and CVRPLIB files. See [Test problems](guide/test-problems.md).
- **Experiments:** `ypea.Experiment` runs many algorithms on many problems in parallel, with reproducible seeds, and resumes interrupted studies. `ypea.Report` gives Wilcoxon, Friedman and Holm tests, CSV, LaTeX, Markdown and HTML tables, figures, a one page HTML report, and the CEC 1e-8 rule (`zero_below`). See [Experiments](guide/experiments.md).
- **Native operators** for GA and SA (`space="native"`): permutation crossovers and moves, operators for real, integer, binary and selection variables, and custom operators. Default results are unchanged. See [Native operators](guide/operators.md).
- `Problem(info=...)` for metadata, `Result.to_dict()` for JSON, and `Result.extra` for information of the algorithm.
- **Tuning:** iterated racing (as in irace) and sweeps of algorithm settings. See [Tuning](tuning.md).
- This documentation site and the `examples` folder.
- Faster evaluation and comparison in the core (5 to 38% less run time), with identical results.
- CMA-ES and IPOP-CMA-ES sample with the symmetric root B D B^T z, so seeded runs do not depend on the eigenvector signs. This changes seeded CMA-ES results once.
- ICA: the weakest empire can no longer win its own colony in the competition.
- [Default values](differences-from-matlab.md#default-values) records which defaults follow their papers, with benchmark evidence.

## 0.3.0 (2026-11-25)

- `ypea.StudioRng`, the random generator of YPEA Studio: `solve(problem, seed=ypea.StudioRng(n))` draws the same numbers as Studio. `ypea.derive_seed` gives the seeds of an experiment.
- Two reference methods: `RandomSearch` and `NelderMead` (with restarts).
- Objective values that are not finite (NaN, inf and -inf) count as the worst.
- Fixes from code audits:
    - IWO: the step size schedule follows the share of the run used, so it ends with the run.
    - CA: at least two solutions update the culture.
    - SA: the Metropolis rule uses the comparison keys, so it is correct with constraints and penalties, and the temperature no longer falls below `final_temp`.
    - IPOP-CMA-ES: restarts on plateaus of infinite values and always grows the population at a restart.
    - Clearer errors for wrong settings.
- Stress tests for every algorithm (one and 30 dimensions, flat, infinite and NaN objectives, problems that are never feasible), and numeric warnings now fail the tests.

## 0.2.0 (2026-11-25)

- All 15 algorithms of YPEA for MATLAB: ABC, ACOR, BA, BBO, CA, CMA-ES, DE, FA, GA, HS, ICA, IWO, PSO, SA and TLBO.
- Four modern baselines: JADE, SHADE, L-SHADE and IPOP-CMA-ES.
- CMA-ES rewritten from the tutorial of Hansen (2016), with the recombination weights and learning rates of its Table 1.
- ABC and IWO use the fitness formula of their papers, and switch to ranks when the problem has constraints.
- Defaults follow the papers: the onlooker count and trial limit of ABC, the initial step size of IPOP-CMA-ES.
- `CA(selection=False)` and `BA(neighborhood="component")` keep the schemes of YPEA for MATLAB.
- [Differences from YPEA for MATLAB](differences-from-matlab.md) lists every change.

## 0.1.1 (2026-11-25)

- A new release to refresh the description on PyPI.

## 0.1.0 (2026-11-25)

- The first release: the core framework (variable types with decoding and encoding, problems with constraints, the comparison rules, evaluation with an exact budget, results and history, callbacks and plots) and the first four algorithms: PSO, DE, GA and SA.
- `max_iter` defaults to None, and 100 iterations apply only when no limit is given.
- Algorithms pickle with their settings only.
