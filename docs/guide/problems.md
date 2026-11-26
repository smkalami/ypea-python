# Problems and variables

## The problem

`ypea.Problem` holds everything about the problem and nothing about the algorithm:

```python
import numpy as np
import ypea

problem = ypea.Problem(
    vars=ypea.Real("x", shape=20, lb=-10, ub=10),     # one variable or a list
    objective=lambda sol: np.sum(sol.x**2),           # gets a decoded Solution
    sense="min",                                      # or "max"
    name="sphere",
    optimum=0.0,                                      # optional: results then report the error
)
```

| Argument | Meaning |
|---|---|
| `vars` | a `Variable` or a list of them, with unique names |
| `objective` | a function of a decoded `Solution` that returns `f` or `(f, g)` (see [Objectives and constraints](objectives.md)) |
| `sense` | `"min"` (default) or `"max"`. `"minimize"`, `"maximization"` and so on also work. |
| `vectorized` | `True` if the objective takes the whole population at once |
| `constraints` | `"feasibility"` (default) or `"penalty"` |
| `penalty` | the penalty coefficient, 1e6 by default |
| `name` | a name for results and reports (default: the name of the objective function) |
| `optimum` | the known optimal value, if any. `Result.error` is then `abs(f - optimum)`. |
| `info` | a dict of free metadata, for example the optimal solution or the source of an instance |

## How variables work

Every variable is stored as a block of numbers in [0, 1], called codes or random keys. The algorithms only see the codes, so every algorithm works with every variable type. A variable knows how many codes it needs (`code_size`), how to turn codes into values (`decode`) and how to turn values back into codes (`encode`).

`decode` works on batches: codes of shape (n, code_size) give values of shape (n, *shape). Indices in permutations, selections and partitions start at 0.

The examples below decode the same five codes with each type:

```python
codes = np.array([[0.7, 0.1, 0.4, 0.9, 0.2]])
```

### Real

`Real(name, shape=None, lb=0.0, ub=1.0)`: continuous values in [lb, ub]. The bounds may be arrays, one per element, and the shape follows them when it is not given.

```python
print(ypea.Real("x", shape=5, lb=-10, ub=10).decode(codes))
```

```text
[[ 4. -8. -2.  8. -6.]]
```

### Integer and Binary

`Integer(name, shape=None, lb=0, ub=1)`: integers in [lb, ub], both included. Every value is equally likely for uniform codes. `Binary(name, shape=None)` is an integer in [0, 1].

```python
print(ypea.Integer("n", shape=5, lb=1, ub=6).decode(codes))
print(ypea.Binary("b", shape=5).decode(codes))
```

```text
[[5 1 3 6 2]]
[[1 0 0 1 0]]
```

### Permutation

`Permutation(name, n, count=None)`: an ordering of 0 to n - 1, from sorting the codes. With `count`, the value is a (count, n) array with one permutation per row.

```python
print(ypea.Permutation("p", n=5).decode(codes))
```

```text
[[1 4 2 0 3]]
```

The smallest code (0.1) belongs to item 1, so 1 comes first.

### Selection and BinarySelection

`Selection(name, n, k, count=None)`: an ordered selection of k distinct items out of n (the first k items of the permutation). `BinarySelection(name, n, k, count=None)`: the same choice as a 0/1 mask of length n with exactly k ones.

```python
print(ypea.Selection("s", n=5, k=2).decode(codes))
print(ypea.BinarySelection("m", n=5, k=2).decode(codes))
```

```text
[[1 4]]
[[0 1 0 0 1]]
```

### Partition and BinaryPartition

`Partition(name, n, k, count=None)`: splits the items 0 to n - 1 into k groups, some of which may be empty. The code has n + k - 1 keys: n items and k - 1 separators, sorted together. The value is a list of k index arrays, so a problem with a partition cannot be vectorized. `BinaryPartition(name, n, k, count=None)` gives the same split as a (k, n) 0/1 assignment matrix, which can be vectorized.

```python
print(ypea.Partition("g", n=4, k=2).decode(codes))
print(ypea.BinaryPartition("g", n=4, k=2).decode(codes))
```

```text
[[array([1]), array([2, 0, 3])]]
[[[0 1 0 0]
  [1 0 1 1]]]
```

Key 4 is the separator. Sorted, the keys are 1, 4, 2, 0, 3, so item 1 is in the first group and items 2, 0 and 3 are in the second.

### FixedSum

`FixedSum(name, shape, target=1.0, axis=None)`: non negative reals with a fixed sum. With `axis=None` all elements sum to `target`. With a 2D shape, `axis=1` fixes the sum of each row and `axis=0` the sum of each column, and `target` may give one value per row or column.

```python
print(ypea.FixedSum("w", shape=5, target=1).decode(codes).round(3))
x = ypea.FixedSum("x", shape=(2, 3), target=[5, 10], axis=1).sample(np.random.default_rng(1))
print(x.round(2), x.sum(axis=1))
```

```text
[[0.304 0.043 0.174 0.391 0.087]]
[[1.59 2.96 0.45]
 [5.63 1.85 2.51]] [ 5. 10.]
```

### Summary

| Type | Decodes to | Typical use |
|---|---|---|
| `Real(name, shape, lb, ub)` | real array in [lb, ub] | continuous optimization |
| `Integer(name, shape, lb, ub)` | integer array in [lb, ub] | integer knapsack, assignment |
| `Binary(name, shape)` | 0/1 array | knapsack |
| `Permutation(name, n, count=None)` | ordering of 0 to n - 1 | TSP, QAP, n-Queens |
| `Selection(name, n, k, count=None)` | k distinct items out of n | feature selection, hub location |
| `BinarySelection(name, n, k, count=None)` | 0/1 mask with k ones | feature selection |
| `Partition(name, n, k, count=None)` | list of k groups of items | clustering, allocation |
| `BinaryPartition(name, n, k, count=None)` | (k, n) 0/1 assignment matrix | allocation |
| `FixedSum(name, shape, target, axis=None)` | non negative reals with fixed totals | portfolio weights, transportation |

`examples/variables.py` prints random samples of every type, including matrices of variables and several permutations or selections at once.

## Random samples

`var.sample(rng)` decodes random codes, which is handy for checking an objective function:

```python
var = ypea.Permutation("tour", n=8)
print(var.sample(np.random.default_rng(1)))
print(var.sample(np.random.default_rng(1), n=2))
```

```text
[2 4 7 5 0 6 3 1]
[[2 4 7 5 0 6 3 1]
 [1 6 4 7 3 0 2 5]]
```

## Several variables

A problem can mix variables of any types. The decoded solution has one attribute per variable, and `sol["name"]` also works. Scalar variables decode to scalars:

```python
def f(sol):
    x, y, z = sol.x, sol.y, sol.z
    return x * y - z**3 - 2 * x - 5 * y + 3 * (x + y) * z**2


problem = ypea.Problem(
    [ypea.Real("x", lb=-2, ub=5), ypea.Real("y", lb=-1, ub=1), ypea.Real("z", lb=3, ub=10)],
    f,
    sense="max",
)
result = ypea.DE(max_iter=200, pop_size=20).solve(problem, seed=1)
print(result.best.obj_value, result.best.solution)
```

```text
790.0 Solution(x=np.float64(5.0), y=np.float64(1.0), z=np.float64(10.0))
```

This is the three variable example of the YPEA for MATLAB documentation (`examples/max_problem.py`).

## Decoding and encoding solutions

A problem decodes whole code vectors and encodes solutions back. `encode` turns a known solution into codes, for example to check it or to compare it with the result of a run:

```python
problem = ypea.Problem(
    [ypea.Permutation("tour", n=4), ypea.Integer("k", lb=0, ub=9)],
    lambda sol: 0.0,
)
code = problem.encode({"tour": [2, 0, 3, 1], "k": 7})
print(code.round(3))
print(problem.decode_one(code))
```

```text
[0.375 0.875 0.125 0.625 0.75 ]
Solution(tour=array([2, 0, 3, 1]), k=np.int64(7))
```

`problem.decode(X)` decodes a batch of codes into a list of solutions, or into one batch solution when the problem is vectorized.

## Your own variable type

A variable type is a subclass of `ypea.Variable` with a `code_size`, a `shape`, and batch `decode` and `encode` methods. Set `array_valued = False` when `decode` returns lists, as `Partition` does. The built in types in `ypea/variables.py` are short examples to start from.
