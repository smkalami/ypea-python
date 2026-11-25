"""Checks of the ICA empire bookkeeping."""

import numpy as np

from ypea.algorithms.ica import ICA


class Watch:
    """Checks the empire arrays after every iteration and records the empire count."""

    def __init__(self):
        self.counts = []

    def __call__(self, alg):
        imps, emp = alg.imperialists, alg.empire_of
        k = len(imps)
        assert len(emp) == len(alg.pop) == alg.pop_size
        assert np.array_equal(emp[imps], np.arange(k))  # each imperialist rules its own empire
        assert emp.min() == 0 and emp.max() == k - 1
        assert alg.colony_counts().sum() == alg.pop_size - k
        if self.counts and k < self.counts[-1]:
            # the imperialist of the collapsed empire is now a colony of another empire
            assert set(self.prev_imps) - set(imps.tolist())
            gone = (set(self.prev_imps) - set(imps.tolist())).pop()
            assert gone not in imps and 0 <= emp[gone] < k
        self.prev_imps = imps.tolist()
        self.counts.append(k)


def test_empires_collapse_consistently_and_run_continues(sphere10):
    w = Watch()
    alg = ICA(pop_size=30, empire_count=6, max_iter=400)
    r = alg.solve(sphere10, seed=1, callbacks=[w])
    assert w.counts[0] == 6 and w.counts[-1] == 1
    assert np.all(np.diff(w.counts) >= -1)  # at most one collapse per iteration
    assert r.iterations == 400 and w.counts.index(1) < 399  # keeps running with one empire


class AtStart:
    def on_start(self, alg):
        self.imps = alg.imperialists.copy()
        self.counts = alg.colony_counts()

    def __call__(self, alg):
        return True


def test_initial_split_is_proportional_and_adds_up(sphere10):
    s = AtStart()
    ICA(pop_size=23, empire_count=4).solve(sphere10, seed=0, callbacks=[s])
    assert np.array_equal(s.imps, np.arange(4)) and s.counts.sum() == 19
    assert np.all(np.diff(s.counts) <= 0)  # stronger imperialists get at least as many colonies


def test_collapse_moves_imperialist_to_another_empire(sphere10):
    s = AtStart()
    alg = ICA(pop_size=10, empire_count=3, max_iter=1)
    alg.solve(sphere10, seed=0, callbacks=[s])
    # empire 2 has one colony, the worst country: after the competition it loses that
    # colony and collapses
    alg.imperialists = np.array([0, 1, 2])
    alg.empire_of = np.array([0, 1, 2, 0, 0, 1, 1, 0, 1, 2])
    c = np.linspace(0, 1, 10)
    alg.compete(c)
    assert np.array_equal(alg.imperialists, [0, 1])
    assert alg.empire_of[2] in (0, 1) and alg.empire_of[9] in (0, 1)
    assert alg.colony_counts().sum() == 8 and len(alg.empire_of) == 10
    alg.imperialists, alg.empire_of = np.array([0]), np.zeros(10, dtype=int)
    alg.compete(c)
    assert np.array_equal(alg.empire_of, np.zeros(10))
