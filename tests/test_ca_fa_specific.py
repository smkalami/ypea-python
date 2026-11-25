"""Checks for CA and FA behavior."""

import numpy as np

from ypea.algorithms.ca import CA
from ypea.algorithms.fa import FA


def _offspring(alg, problem):
    """Run one more iteration and return (parents, situational before it, offspring codes)."""
    alg.solve(problem, seed=0)
    seen = []
    evaluate = alg.evaluate
    alg.evaluate = lambda X: seen.append(X.copy()) or evaluate(X)
    parents = alg.pop.X.copy()
    s = getattr(alg, "situational", None)
    alg.iterate()
    return parents, s, seen[0]


def test_ca_steps_each_component_toward_situational(sphere10):
    # MATLAB added the step of component j to the whole vector (review item A4)
    alg = CA(influence_type=2, max_iter=1, pop_size=20)
    parents, s, X = _offspring(alg, sphere10)
    s = s.X
    toward = np.sign(X - parents) == np.sign(s - parents)
    assert toward[parents != s].all()


def test_ca_culture_covers_accepted_and_keeps_best(sphere10):
    alg = CA(max_iter=20, pop_size=20)
    r = alg.solve(sphere10, seed=1)
    acc = alg.pop.X[: alg._accept_count]
    assert (alg.xmin <= acc.min(0) + 1e-12).all() and (alg.xmax >= acc.max(0) - 1e-12).all()
    assert alg.situational.F[0, 0] == r.best.obj_value


def test_fa_brightest_makes_small_random_walk(sphere10):
    alg = FA(max_iter=1, pop_size=10)
    parents, _, X = _offspring(alg, sphere10)
    step = np.abs(X[0] - parents[0])  # the population is sorted, so 0 is the brightest
    assert 0 < step.max() <= alg._alpha / alg.alpha_damp * alg.delta
    assert alg.nfe == 30  # pop_size evaluations per iteration
