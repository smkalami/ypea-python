# YPEA for Python

Yarpiz Evolutionary Algorithms (YPEA) is a toolbox for defining an optimization problem once and solving it with any of its metaheuristics. This is the Python version of [YPEA for MATLAB](https://github.com/smkalami/ypea), redesigned and with the known defects fixed.

**Status:** early development (0.2.0). All 15 algorithms of YPEA for MATLAB and four modern baselines work. Benchmark suites, test problems and the experiment runner are being added.

## Install

```
pip install ypea            # core, needs only NumPy
pip install "ypea[plot]"    # with matplotlib for plots
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

## Algorithms

The 15 algorithms of YPEA for MATLAB, with their known defects fixed:

| Class | Algorithm |
|---|---|
| `ABC` | Artificial Bee Colony |
| `ACOR` | Ant Colony Optimization for Continuous Domains |
| `BA` | Bees Algorithm (standard and probabilistic) |
| `BBO` | Biogeography-based Optimization |
| `CA` | Cultural Algorithm |
| `CMAES` | Covariance Matrix Adaptation Evolution Strategy |
| `DE` | Differential Evolution (rand, best, current-to-best, rand-to-best, bin and exp) |
| `FA` | Firefly Algorithm |
| `GA` | Genetic Algorithm |
| `HS` | Harmony Search |
| `ICA` | Imperialist Competitive Algorithm |
| `IWO` | Invasive Weed Optimization |
| `PSO` | Particle Swarm Optimization |
| `SA` | Simulated Annealing |
| `TLBO` | Teaching-Learning-based Optimization |

Four modern baselines for benchmark comparisons:

| Class | Algorithm |
|---|---|
| `JADE` | Adaptive Differential Evolution with external archive |
| `SHADE` | Success-History based Adaptive Differential Evolution |
| `LSHADE` | SHADE with Linear Population Size Reduction |
| `IPOPCMAES` | CMA-ES with restarts and increasing population size |

See `docs/differences-from-matlab.md` and `docs/new-algorithms.md` for details.

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

The objective may return `(f, g)`, where each value in `g` is satisfied when `g <= 0`. By default, solutions are compared with Deb's feasibility rules. Use `constraints="penalty"` for a penalty method.

```python
def evaluate(sol):
    x, y = sol.x
    return (x - 1) ** 2 + (y - 2) ** 2, [x + y - 2]

problem = ypea.Problem(ypea.Real("x", shape=2, lb=-5, ub=5), evaluate)
```

## Citing

For this Python version (see `CITATION.cff`):

Mostapha Kalami Heris, YPEA: Yarpiz Evolutionary Algorithms for Python, version 0.2.0, 2026. https://github.com/smkalami/ypea-python

For the original MATLAB toolbox:

Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.

## License

MIT
