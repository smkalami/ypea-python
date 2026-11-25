# YPEA for Python

Yarpiz Evolutionary Algorithms (YPEA) is a toolbox for defining an optimization problem once and solving it with any of its metaheuristics. This is the Python version of [YPEA for MATLAB](https://github.com/smkalami/ypea), redesigned and with the known defects fixed.

**Status:** early development (0.1.0.dev0). The core and the first algorithms work, and more are being added.

## Install

```
pip install ypea            # core, needs only NumPy
pip install ypea[plot]      # with matplotlib for plots
```

Python 3.12 or newer.

## Example

Find 20 real numbers in [-10, 10] that minimize the sphere function:

```python
import ypea

problem = ypea.Problem(
    vars=ypea.Real("x", shape=20, lb=-10, ub=10),
    objective=lambda sol: ypea.benchmarks.sphere(sol.x),
)

pso = ypea.PSO(max_iter=100, pop_size=100, w=0.5, wdamp=1, c1=1, c2=2)
result = pso.solve(problem, seed=42)

print(result.best.solution.x)
print(result.best.obj_value)
```

Any other algorithm solves the same problem:

```python
de = ypea.DE(strategy="DE/best/2/bin", max_iter=1000, pop_size=20,
             beta_min=0.1, beta_max=0.9, crossover_prob=0.1)
result = de.solve(problem, seed=42)
```

## Decision variables

All variable types are encoded as numbers in [0, 1], so every algorithm can solve every problem. Decoded indices are 0 based.

| Type | Decodes to | Typical use |
|---|---|---|
| `Real(name, shape, lb, ub)` | real array in [lb, ub] | continuous optimization |
| `Integer(name, shape, lb, ub)` | integer array in [lb, ub] | generalized knapsack |
| `Binary(name, shape)` | 0/1 array | knapsack |
| `Permutation(name, n, count=None)` | permutation of 0..n-1 (one per row) | TSP, QAP, n-Queens |
| `Selection(name, n, k, count=None)` | k distinct items out of n | feature selection |
| `BinarySelection(name, n, k, count=None)` | 0/1 mask with k ones | feature selection |
| `Partition(name, n, k, count=None)` | list of k groups of items | vehicle routing, allocation |
| `BinaryPartition(name, n, k, count=None)` | (k, n) 0/1 assignment matrix | hub location allocation |
| `FixedSum(name, shape, target, axis=None)` | non negative reals with a fixed total, row sums or column sums | transportation |

A problem can mix several variables. The decoded solution has one attribute per variable (`sol.x`, `sol.tour`).

## Constraints

The objective may return `(f, g)`, where each value in `g` is satisfied when `g <= 0`. By default, solutions are compared with Deb's feasibility rules. Use `constraint_handling="penalty"` for a penalty method.

```python
def evaluate(sol):
    x, y = sol.x
    return (x - 1) ** 2 + (y - 2) ** 2, [x + y - 2]

problem = ypea.Problem(ypea.Real("x", shape=2, lb=-5, ub=5), evaluate)
```

## Citing

Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox (YPEA), Yarpiz, 2020.

## License

MIT
