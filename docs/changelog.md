# Changelog

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
