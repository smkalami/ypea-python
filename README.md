# YPEA for Python

Yarpiz Evolutionary Algorithms (YPEA) is a toolbox for defining an optimization problem once and solving it with any of its 21 metaheuristics. This is the Python version of [YPEA for MATLAB](https://github.com/smkalami/ypea), redesigned and with the known defects fixed.

**Status:** early development (0.5.0).

## Install

```
pip install ypea              # core, needs only NumPy
pip install "ypea[plot]"      # with matplotlib for plots
pip install "ypea[stats]"     # with SciPy for the statistical tests of experiments
```

Python 3.12 or newer. To work on YPEA itself, clone the repository and run `pip install -e ".[dev]"`.

## Example

Find 20 real numbers in [-10, 10] that minimize the sphere function:

```python
import ypea

problem = ypea.Problem(
    vars=ypea.Real("x", shape=20, lb=-10, ub=10),
    objective=lambda sol: ypea.benchmarks.sphere(sol.x),
)

pso = ypea.PSO(max_iter=100, pop_size=100, w=0.5, wdamp=1, c1=1, c2=2)
result = pso.solve(problem, seed=1)

print(result.best.solution.x)
print(result.best.obj_value)
```

Any other algorithm solves the same problem, for example `ypea.LSHADE(max_nfe=20_000).solve(problem, seed=1)`.

## Features

- **21 algorithms:** the 15 of YPEA for MATLAB (ABC, ACOR, BA, BBO, CA, CMA-ES, DE, FA, GA, HS, ICA, IWO, PSO, SA, TLBO), the modern baselines JADE, SHADE, L-SHADE and IPOP-CMA-ES, and the reference methods Random Search and Nelder-Mead.
- **Every variable type for every algorithm:** real, integer, binary, permutation, selection, partition and fixed sum variables, all encoded in [0, 1]. GA and SA also have native operators for permutations, selections, integers and reals (`space="native"`).
- **Constraints built in:** the objective may return `(f, g)` with `g <= 0`. Deb's feasibility rules by default, or a penalty.
- **Reproducible and exact:** the same seed gives the same result, the evaluation budget is never exceeded, and every improvement is recorded. `seed=ypea.StudioRng(n)` draws the same numbers as YPEA Studio.
- **Fast:** vectorized objectives and parallel evaluation.
- **Benchmarks:** 26 classic functions with optional shift and rotation, and CEC 2017 and CEC 2022, checked against the official C code.
- **Test problems:** engineering design, TSP, knapsack, QAP, n-Queens, vehicle routing, generalized assignment, hub location and transportation, with readers for the standard instance libraries.
- **Experiments:** many algorithms on many problems, in parallel and resumable, with Wilcoxon, Friedman and Holm tests, CSV, LaTeX, Markdown and HTML tables, figures and a one page HTML report.

## Documentation

The documentation site is at [smkalami.github.io/ypea-python](https://smkalami.github.io/ypea-python/), with the user guide, one page per algorithm and the API reference. Its sources are in the [`docs`](https://github.com/smkalami/ypea-python/blob/main/docs/index.md) folder. The [`examples`](https://github.com/smkalami/ypea-python/tree/main/examples) folder has complete scripts. If you know the MATLAB toolbox, start with the [differences from YPEA for MATLAB](https://github.com/smkalami/ypea-python/blob/main/docs/differences-from-matlab.md).

## Citing

For this Python version (see `CITATION.cff`):

Mostapha Kalami Heris, YPEA: Yarpiz Evolutionary Algorithms for Python, version 0.5.0, 2026. https://github.com/smkalami/ypea-python

For the original MATLAB toolbox:

Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.

## License

MIT
