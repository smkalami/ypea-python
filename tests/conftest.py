import sys
from pathlib import Path

import pytest

import ypea

sys.path.insert(0, str(Path(__file__).parent))

from problems import constrained_obj, mixed_obj, sphere_obj  # noqa: E402


@pytest.fixture
def sphere10():
    return ypea.Problem(ypea.Real("x", shape=10, lb=-10, ub=10), sphere_obj, name="sphere")


@pytest.fixture
def constrained():
    return ypea.Problem(ypea.Real("x", shape=2, lb=-5, ub=5), constrained_obj, name="constrained")


@pytest.fixture
def mixed():
    return ypea.Problem([
        ypea.Real("r", shape=3, lb=-1, ub=1),
        ypea.Integer("i", shape=2, lb=0, ub=5),
        ypea.Binary("b", shape=3),
        ypea.Permutation("p", n=5),
        ypea.Selection("s", n=6, k=2),
        ypea.BinarySelection("bs", n=6, k=2),
        ypea.Partition("q", n=6, k=3),
        ypea.BinaryPartition("bp", n=4, k=2),
        ypea.FixedSum("fs", shape=3, target=1),
    ], mixed_obj, name="mixed")
