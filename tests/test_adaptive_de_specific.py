"""Checks for JADE, SHADE and L-SHADE."""

import numpy as np
import pytest

from ypea.algorithms.adaptive_de import JADE, LSHADE, SHADE


def test_lshade_population_shrinks_linearly_to_min(sphere10):
    sizes = []
    alg = LSHADE(max_nfe=5_000)
    alg.solve(sphere10, seed=0, callbacks=[lambda a: sizes.append((a.nfe, len(a.pop)))])
    n0 = 18 * 10
    for nfe, n in sizes:
        assert n == round(n0 + (4 - n0) * nfe / 5_000)
    assert sizes[-1] == (5_000, 4)
    assert len(alg._archive) <= round(2.6 * 4)


def test_lshade_without_max_nfe_reaches_min_at_max_iter(sphere10):
    alg = LSHADE(max_iter=10, pop_size=50)
    alg.solve(sphere10, seed=0)
    assert len(alg.pop) == 4


@pytest.mark.parametrize("cls", [JADE, SHADE, LSHADE])
def test_memory_and_samples_in_range(cls, sphere10):
    alg = cls(max_nfe=3_000, pop_size=30)
    alg.solve(sphere10, seed=1)
    if cls is JADE:
        mem_f, mem_cr = np.array([alg.mu_f]), np.array([alg.mu_cr])
    else:
        mem_f, mem_cr = alg.m_f, alg.m_cr
    assert np.all((mem_f > 0) & (mem_f <= 1))
    cr = mem_cr[~np.isnan(mem_cr)]
    assert np.all((cr >= 0) & (cr <= 1))
    assert cls is LSHADE or not np.isnan(mem_cr).any()  # only L-SHADE has the terminal value
    F, CR = alg._sample(10_000)
    assert np.all((F > 0) & (F <= 1)) and np.all((CR >= 0) & (CR <= 1))


def test_lshade_terminal_cr_gives_zero(sphere10):
    alg = LSHADE(max_iter=1, pop_size=20)
    alg.solve(sphere10, seed=0)
    alg.m_cr[:] = np.nan
    _, CR = alg._sample(100)
    assert np.all(CR == 0)
    alg._update_memory(np.array([0.5]), np.array([0.9]), np.array([1.0]))
    assert np.isnan(alg.m_cr).all()  # terminal stays terminal


def test_settings_are_validated():
    with pytest.raises(ValueError):
        JADE(pop_size=3)
    with pytest.raises(ValueError):
        LSHADE(pop_size=3)
    with pytest.raises(ValueError):
        SHADE(memory_size=0)
