"""Every type of decision variable, with random samples of its decoded values.

A Python version of doc_ypea_var.m of YPEA for MATLAB. Each variable is stored as codes in
[0, 1], and ``decode`` turns codes into values. ``sample`` decodes random codes. Indices
are 0 based. Run it as ``python variables.py``.
"""

import numpy as np

import ypea

rng = np.random.default_rng(1)
np.set_printoptions(precision=3, suppress=True)


def show(var):
    print(f"{var!r}, {var.code_size} codes")
    print(var.sample(rng), "\n")


# real values: a scalar in [0, 1], 10 values in [-5, 5], and one bound per value
show(ypea.Real("x"))
show(ypea.Real("y", shape=10, lb=-5, ub=5))
show(ypea.Real("z", lb=[-1, 3, -10], ub=[1, 5, 2]))
show(ypea.Real("X", shape=(3, 5), lb=-3, ub=8))
show(ypea.Real("A", lb=[[-5, 1, 3], [0, 0, -8]], ub=[[-2, 4, 7], [1, 5, 8]]))

# integers, both bounds included
show(ypea.Integer("n"))  # 0 or 1
show(ypea.Integer("m", shape=8, lb=-3, ub=6))
show(ypea.Integer("p", lb=[-1, 3, -10], ub=[1, 5, 2]))

# binary: a 3 by 6 matrix of 0 and 1
show(ypea.Binary("b", shape=(3, 6)))

# permutations of 0..n-1, one or several (one per row)
show(ypea.Permutation("tour", n=8))
show(ypea.Permutation("p", n=8, count=3))

# selections: 4 distinct items out of 10, as indices or as a 0/1 mask
show(ypea.Selection("a", n=10, k=4))
show(ypea.BinarySelection("b", n=10, k=4))
show(ypea.Selection("c", n=10, k=4, count=3))
show(ypea.BinarySelection("d", n=10, k=4, count=3))

# partitions: 10 items in 3 groups (some may be empty), as lists or as a 0/1 matrix
show(ypea.Partition("a", n=10, k=3))
show(ypea.BinaryPartition("b", n=10, k=3))
show(ypea.Partition("c", n=10, k=3, count=2))

# fixed sums: all elements, each row, or each column sum to 5
for axis in (None, 1, 0):
    var = ypea.FixedSum("x", shape=(3, 4), target=5, axis=axis)
    value = var.sample(rng)
    print(f"{var!r}, axis={axis}")
    print(value.round(3))
    print("sums:", value.sum() if axis is None else value.sum(axis=axis).round(6), "\n")

# several variables in one problem: the decoded solution has one attribute per variable
problem = ypea.Problem(
    [ypea.Integer("a", shape=3, lb=1, ub=6), ypea.Real("x", shape=3, lb=-2, ub=2),
     ypea.Integer("m", lb=0, ub=10)],
    lambda sol: 0.0,
)
codes = rng.random((1, problem.code_size))
sol = problem.decode(codes)[0]
print("decoded solution:", sol)

# encode turns values back into codes, so known solutions can join a run
var = ypea.Permutation("tour", n=5)
codes = var.encode([[3, 0, 4, 1, 2]])
print("codes of [3, 0, 4, 1, 2]:", codes.round(2), "decode back:", var.decode(codes))
