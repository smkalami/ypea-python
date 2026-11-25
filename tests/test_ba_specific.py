import numpy as np
import pytest

from ypea.algorithms.ba import BA


def test_derived_counts_round_half_away_from_zero():
    ba = BA(pop_size=45)
    assert ba.selected_site_count == 23 and ba.elite_site_count == 9
    assert ba.selected_site_bee_count == 5 and ba.elite_site_bee_count == 10
    assert ba.recruited_bee_count == 5


def test_type_aliases_and_validation():
    assert BA(type="std").type == "standard" and BA(type="PROB").type == "probabilistic"
    with pytest.raises(ValueError):
        BA(type="other")
    with pytest.raises(ValueError):
        BA(selected_site_ratio=0)


@pytest.mark.parametrize("kind", ["standard", "probabilistic"])
def test_small_and_odd_settings_run(kind, sphere10):
    for n in (1, 2, 3):
        r = BA(type=kind, pop_size=n, max_nfe=50).solve(sphere10, seed=0)
        assert np.isfinite(r.best.obj_value)
