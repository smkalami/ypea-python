import numpy as np
import pytest

from ypea.algorithms.hs import HS
from ypea.algorithms.tlbo import TLBO


def test_hs_defaults_and_alias():
    hs = HS()
    assert hs.pop_size == hs.hms == 10
    hs.hms = 7
    assert hs.pop_size == 7


def test_hs_bandwidth_is_damped(sphere10):
    hs = HS(max_iter=5, fret_width=0.1, fret_width_damp=0.5)
    hs.solve(sphere10, seed=0)
    assert np.isclose(hs.bandwidth, 0.1 * 0.5**5)


def test_tlbo_two_evaluations_per_learner(sphere10):
    r = TLBO(pop_size=10, max_iter=3).solve(sphere10, seed=0)
    assert r.nfe == 10 + 3 * 2 * 10


def test_tlbo_needs_two_learners():
    with pytest.raises(ValueError):
        TLBO(pop_size=1)
