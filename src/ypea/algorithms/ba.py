"""Bees Algorithm (BA)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_real

_TYPES = {"standard": "standard", "std": "standard", "probabilistic": "probabilistic", "prob": "probabilistic"}


def _round(x):
    # MATLAB round (half away from zero), so derived counts match YPEA for MATLAB
    return int(np.floor(x + 0.5))


class BA(Algorithm):
    """Bees Algorithm (Pham et al., 2005, 2006).

    Each solution is a site (a flower patch) visited by a scout bee. The sites are kept
    sorted from best to worst.

    Standard mode (Pham et al.): the best ``selected_site_count`` sites are searched by
    recruited bees, ``elite_site_bee_count`` bees for each of the ``elite_site_count``
    elite sites and ``selected_site_bee_count`` bees for each other selected site. Every
    recruited bee is drawn uniformly from the patch around its site, a box of half width
    r in every component. A site moves to its best recruited bee only if that bee is
    strictly better. All other sites are replaced by new random scouts.

    Probabilistic mode (as in YPEA for MATLAB): every site gets a relative score d with
    mean 1. A site is abandoned for a new random scout with probability 0.6 if
    d < 0.9, 0.2 if 0.9 <= d < 0.95, 0.05 if 0.95 <= d < 1.15 and 0 otherwise. A site
    that is kept gets ceil(d * recruited_bee_count) recruited bees, and each of them
    changes one random component of the site by r * U(-1, 1). The score is
    d = n * p, where p are the rank based selection probabilities (p proportional to
    exp(-r) with r from 0 for the best to 1 for the worst). So d goes from about 1.58
    for the best site to about 0.58 for the worst, and it does not depend on the scale
    or the sign of the objective.

    Parameters
    ----------
    pop_size : number of scout bees (``scout_bee_count``), which is also the number of sites
    type : "standard" (or "std") or "probabilistic" (or "prob")
    selected_site_ratio : selected sites as a fraction of the scout bees (standard mode)
    selected_site_bee_ratio : bees per selected site as a fraction of the scout bees (standard mode)
    elite_site_ratio : elite sites as a fraction of the selected sites (standard mode)
    elite_site_bee_ratio : bees per elite site as a multiple of the bees per selected site (standard mode)
    recruited_bee_ratio : base number of recruited bees as a fraction of the scout bees
        (probabilistic mode)
    dance_radius : initial neighborhood size r (code units)
    dance_radius_damp : r is multiplied by this after every iteration
    neighborhood : how a recruited bee leaves its site in the standard mode. "patch"
        (default, the paper) draws every component uniformly from the box of half width
        r. "component" changes one random component by r * U(-1, 1), as in YPEA for
        MATLAB, which works much better on problems with many components.

    Derived counts are rounded half away from zero, as in MATLAB. Bee counts per site are
    at least 1, so small populations still search their sites.

    Differences from the paper: the 2006 paper keeps the neighborhood size fixed. The
    damping of r comes from YPEA for MATLAB (``dance_radius_damp=1`` gives a fixed size).
    The probabilistic mode and its thresholds come from YPEA for MATLAB, with a rank
    based score in place of 1 / f.

    References: Pham, Ghanbarzadeh, Koc, Otri, Rahim and Zaidi (2005), The Bees
    Algorithm, Technical Note, Manufacturing Engineering Centre, Cardiff University.
    Pham, Ghanbarzadeh, Koc, Otri, Rahim and Zaidi (2006), The Bees Algorithm: a novel
    tool for complex optimisation problems, IPROMS 2006.
    """

    name = "Bees Algorithm"
    short_name = "BA"

    def __init__(self, *, type="standard", selected_site_ratio=0.5, selected_site_bee_ratio=0.1,  # noqa: A002
                 elite_site_ratio=0.4, elite_site_bee_ratio=2.0, recruited_bee_ratio=0.1,
                 dance_radius=0.1, dance_radius_damp=0.99, neighborhood="patch", **kwargs):
        self.neighborhood = neighborhood
        self.type = _TYPES.get(str(type).lower(), type)
        self.selected_site_ratio = selected_site_ratio
        self.selected_site_bee_ratio = selected_site_bee_ratio
        self.elite_site_ratio = elite_site_ratio
        self.elite_site_bee_ratio = elite_site_bee_ratio
        self.recruited_bee_ratio = recruited_bee_ratio
        self.dance_radius = dance_radius
        self.dance_radius_damp = dance_radius_damp
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(self.type in ("standard", "probabilistic"), "type must be 'standard' or 'probabilistic'")
        check(self.neighborhood in ("patch", "component"), "neighborhood must be 'patch' or 'component'")
        for name in ("selected_site_ratio", "selected_site_bee_ratio", "elite_site_ratio", "recruited_bee_ratio"):
            v = getattr(self, name)
            check(is_real(v, 0, 1) and v > 0, f"{name} must be in (0, 1]")
        check(is_real(self.elite_site_bee_ratio, 0) and self.elite_site_bee_ratio > 0,
              "elite_site_bee_ratio must be positive")
        check(is_real(self.dance_radius, 0), "dance_radius must be non negative")
        check(is_real(self.dance_radius_damp, 0) and self.dance_radius_damp > 0, "dance_radius_damp must be positive")

    # derived counts (as in YPEA for MATLAB)

    @property
    def scout_bee_count(self):
        return self.pop_size

    @property
    def selected_site_count(self):
        return _round(self.selected_site_ratio * self.scout_bee_count)

    @property
    def selected_site_bee_count(self):
        return max(1, _round(self.selected_site_bee_ratio * self.scout_bee_count))  # at least one bee

    @property
    def elite_site_count(self):
        return _round(self.elite_site_ratio * self.selected_site_count)

    @property
    def elite_site_bee_count(self):
        return _round(self.elite_site_bee_ratio * self.selected_site_bee_count)

    @property
    def recruited_bee_count(self):
        return max(1, _round(self.recruited_bee_ratio * self.scout_bee_count))  # at least one bee

    def initialize(self):
        self.pop = self.sort(self.random_population())
        self._r = self.dance_radius

    def iterate(self):
        if self.type == "standard":
            m, e = self.selected_site_count, self.elite_site_count
            counts = np.r_[np.full(e, self.elite_site_bee_count), np.full(m - e, self.selected_site_bee_count)]
            self._recruit(counts, patch=self.neighborhood == "patch")
            scouts = np.arange(m, self.scout_bee_count)
        else:
            d = len(self.pop) * self.selection_probs(self.pop)
            reject_prob = np.select([d < 0.9, d < 0.95, d < 1.15], [0.6, 0.2, 0.05], 0.0)
            keep = self.rng.random(len(d)) >= reject_prob
            counts = np.where(keep, np.ceil(d * self.recruited_bee_count), 0).astype(int)
            self._recruit(counts, patch=False)
            scouts = np.flatnonzero(~keep)
        if len(scouts):
            self._set(scouts, self.random_population(len(scouts)))
        self.pop = self.sort(self.pop)
        self._r *= self.dance_radius_damp

    def _recruit(self, counts, patch):
        """Send counts[i] bees to site i and move each site to its best bee if that is strictly better."""
        parents = np.repeat(np.arange(len(counts)), counts)
        if not len(parents):
            return
        X = self.pop.X[parents]
        if patch:
            X = X + self._r * self.rng.uniform(-1, 1, X.shape)
        else:
            rows = np.arange(len(X))
            X[rows, self.rng.integers(self.dim, size=len(X))] += self._r * self.rng.uniform(-1, 1, len(X))
        bees = self.evaluate(X)
        order = self.argsort(bees)
        sites, first = np.unique(parents[order], return_index=True)
        best = bees[order[first]]
        win = self.better(best, self.pop[sites])
        if win.any():
            self._set(sites[win], best[np.flatnonzero(win)])

    def _set(self, idx, new):
        self.pop.X[idx], self.pop.F[idx], self.pop.V[idx] = new.X, new.F, new.V
