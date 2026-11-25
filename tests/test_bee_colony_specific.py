"""Checks of ABC's own behavior."""

import numpy as np

from ypea.algorithms.bee_colony import ABC


def test_abc_changes_one_component_and_resets_trials(sphere10):
    alg = ABC(max_iter=3, pop_size=10, onlooker_count=10)
    alg.solve(sphere10, seed=0)
    assert alg._limit == 10 * 10  # onlooker_count * d, as in Karaboga (2005, Table 2)
    before = alg.pop.X.copy()
    new = alg.search(np.arange(10))
    assert np.all((new.X != before).sum(axis=1) <= 1)
    alg.trials[:] = 5
    alg.iterate()
    assert alg.trials.max() >= 6  # failed sources counted on from 5
    assert (alg.trials == 0).any()  # some sources improved and were reset


def test_paper_defaults_follow_the_population_size():
    alg = ABC(max_iter=1, pop_size=12)
    from problems import sphere_obj

    import ypea
    alg.solve(ypea.Problem(ypea.Real("x", shape=5), sphere_obj), seed=0)
    assert alg._onlookers == 12 and alg._limit == 12 * 5
