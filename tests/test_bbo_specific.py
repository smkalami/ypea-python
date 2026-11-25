import numpy as np
import pytest

from ypea.algorithms.bbo import BBO


@pytest.mark.parametrize("keep_rate", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_edge_settings_run(sphere10, n, keep_rate):
    r = BBO(pop_size=n, keep_rate=keep_rate, max_nfe=60).solve(sphere10, seed=0)
    assert r.nfe == 60 and np.isfinite(r.best.obj_value)

