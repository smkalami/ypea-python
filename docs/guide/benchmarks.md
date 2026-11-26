# Benchmarks

`ypea.benchmarks` has the classic test functions and the CEC 2017 and CEC 2022 suites. Every function works on one point with shape (d,) and on a batch with shape (..., d), and every ready made problem is vectorized.

## Classic functions

26 functions, each with its standard search bounds, its optimum value and the location of its optimum:

| Function | Bounds | Function | Bounds |
|---|---|---|---|
| `sphere` | [-100, 100] | `levy` | [-10, 10] |
| `ellipsoid` | [-100, 100] | `zakharov` | [-5, 10] |
| `sum_squares` | [-10, 10] | `step` | [-100, 100] |
| `schwefel_1_2` | [-100, 100] | `quartic` | [-1.28, 1.28] |
| `schwefel_2_21` | [-100, 100] | `alpine1` | [-10, 10] |
| `schwefel_2_22` | [-10, 10] | `styblinski_tang` | [-5, 5] |
| `schwefel_2_26` | [-500, 500] | `dixon_price` | [-10, 10] |
| `rosenbrock` | [-30, 30] | `michalewicz` | [0, pi] |
| `rastrigin` | [-5.12, 5.12] | `salomon` | [-100, 100] |
| `ackley` | [-32.768, 32.768] | `weierstrass` | [-0.5, 0.5] |
| `griewank` | [-600, 600] | `happy_cat` | [-100, 100] |
| `bent_cigar` | [-100, 100] | `hgbat` | [-100, 100] |
| `discus` | [-100, 100] | `expanded_schaffer_f6` | [-100, 100] |

All have the optimum value 0, except Styblinski Tang (-39.166... d) and Michalewicz, whose optimum is computed numerically for each dimension (-9.66015 for d = 10). `quartic_noisy` adds uniform noise to `quartic` and is not part of the suite. The docstring of each function gives its formula and source (Yao, Liu and Lin 1999, Jamil and Yang 2013, and the CEC technical reports).

Use a function directly in your own objective:

```python
import numpy as np
import ypea

print(ypea.benchmarks.rastrigin(np.zeros(5)), ypea.benchmarks.ackley(np.ones((3, 5))))
```

```text
0.0 [3.62538494 3.62538494 3.62538494]
```

Or get a ready made `Problem`, with the optimum set so results report the error:

```python
problem = ypea.benchmarks.problem("rastrigin", 10)
print(problem.name, problem.optimum, problem.vars)
```

```text
rastrigin_d10 0.0 [Real('x', shape=(10,))]
```

### Shift and rotation

Many functions have their optimum at the origin, in the middle of the search space, which some algorithms find too easily. `shift=True` moves the optimum to a random point in the central 80 percent of the bounds, and `rotate=True` rotates the function around its optimum with a random orthogonal matrix, so the variables are no longer separable. Both are fixed by `seed`:

```python
problem = ypea.benchmarks.problem("rastrigin", 10, shift=True, rotate=True, seed=3)
print(problem.name)
print(problem.objective(ypea.Solution(x=problem.info["x_opt"])))  # the value at the optimum
```

```text
rastrigin_d10_shifted_rotated_s3
0.0
```

`info["x_opt"]` is the optimum, and `info["shift"]` and `info["rotate"]` are the vector and the matrix. Michalewicz cannot be shifted or rotated, because its optimum is known only inside the standard bounds.

`ypea.benchmarks.classic(dim)` returns the whole suite as a list of problems, and `ypea.benchmarks.names()` the names. With `shift` or `rotate`, the suite leaves out Michalewicz.

## CEC 2017

The 30 functions of the CEC 2017 competition on single objective bound constrained optimization (Awad, Ali, Liang, Qu and Suganthan, 2016): unimodal (F1 to F3), simple multimodal (F4 to F10), hybrid (F11 to F20) and composition functions (F21 to F30). The search range is [-100, 100] in every dimension, and function F has the optimal value 100 F.

```python
problems = ypea.benchmarks.cec2017(10)
print(len(problems), problems[0].name, problems[0].optimum)
```

```text
29 cec2017_f1_d10 100.0
```

| Dimension | Functions |
|---|---|
| 2 | F1 to F10 and F23 to F28 |
| 10, 30, 50, 100 | F1 to F30 |
| 20 | F1 to F10 and F20 to F28 |

These are the dimensions and functions for which the official data exists. F2 was removed from the competition because it is numerically unstable. It is left out of `cec2017(dim)` by default, and `include_f2=True` or `functions=[2]` adds it. `cec2017(dim, functions=[1, 4, 5])` gives a subset, `cec2017_problem(f, dim)` one problem, and `cec2017_function(f, dim)` the bare array function.

## CEC 2022

The 12 functions of the CEC 2022 competition on single objective bound constrained optimization (Kumar, Price, Mohamed, Hadi and Suganthan, 2021), at dimensions 10 and 20. All but F6, F7 and F8 also exist at dimension 2. The search range is [-100, 100], and the optimal values are 300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600 and 2700.

```python
problems = ypea.benchmarks.cec2022(20)
print([p.optimum for p in problems][:4])
```

```text
[300.0, 400.0, 600.0, 800.0]
```

`cec2022_problem(f, dim)` and `cec2022_function(f, dim)` work as for CEC 2017.

## How the CEC suites were checked

Both suites are ports of the official C code (`cec17_test_func.cpp` and `cec22_test_func.cpp`) with the official shift vectors, rotation matrices and shuffle data. The tests check them in three ways:

1. **Against the official code.** The official C code was compiled and run on a throwaway machine, at 8 points for every function and dimension: 6 points spread over the search space, the shift vector and a point next to it. The Python values match these reference values to a relative tolerance of 1e-10. CEC 2017 F2 is the exception: the official code no longer evaluates it, so it is checked only by the other two ways.
2. **At the optimum.** Every function gives its optimal value at the location in `info["x_opt"]`.
3. **By hand.** Several functions (for example CEC 2017 F5, F6 and F11, and CEC 2022 F10) are also written out a second time from the formulas of the technical reports, and a batch of points gives the same values as the points one by one.

Where the C code differs from the technical report, the port follows the C code, because the published results were computed with it. For example, CEC 2017 F6 and CEC 2022 F3 (Schaffer F7) use the shifted point without the rotation, and the minimum of CEC 2017 F9 is not at the shift vector. The module docstrings list every such difference.

## The CEC 1e-8 rule

The CEC rules count an error below 1e-8 as 0. Experiment reports keep the raw values by default. Set `zero_below` on the report before you compare your tables with published CEC results:

```python
from ypea.experiment import Experiment

exp = Experiment({"DE": ypea.DE(), "L-SHADE": ypea.LSHADE()},
                 ypea.benchmarks.cec2017(10, functions=[1, 3]), runs=3, max_nfe=2_000)
report = exp.run().report()
report.zero_below = 1e-8
print(report.values("L-SHADE", "cec2017_f1_d10").shape)
```

```text
(3,)
```

The CEC 2017 rules also fix the budget (10 000 D evaluations), the number of runs (51) and the points at which the error is recorded. `Experiment` records the error at these points by default (see [Experiments](experiments.md)).

## Data and license

The CEC data files ship inside the package, unchanged, in a compressed NumPy format, with a notice that credits the organizers. The data was created by the organizers of the competitions and comes from their repositories (github.com/P-N-Suganthan). Please cite the technical report of a suite when you publish results obtained with it:

- N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu and P. N. Suganthan, "Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on Single Objective Bound Constrained Real-Parameter Numerical Optimization", Technical Report, Nanyang Technological University, Singapore, 2016.
- A. Kumar, K. V. Price, A. W. Mohamed, A. A. Hadi and P. N. Suganthan, "Problem Definitions and Evaluation Criteria for the CEC 2022 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization", Technical Report, Nanyang Technological University, Singapore, 2021.
