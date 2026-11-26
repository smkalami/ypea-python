"""CMA-ES with restarts and increasing population size (IPOP-CMA-ES)."""

from __future__ import annotations

import numpy as np

from ..algorithm import check, is_int, is_real
from .cmaes import CMAES


def _span(values):
    """Range of the values. Equal values (including a plateau of inf) have range 0."""
    if np.all(values == values[0]):
        return 0.0
    with np.errstate(invalid="ignore"):
        s = np.ptp(values)
    return s if np.isfinite(s) else np.inf


class IPOPCMAES(CMAES):
    """IPOP-CMA-ES: CMA-ES that restarts with a larger population when it stops making progress.

    Each run is the ``CMAES`` of this package. When a restart criterion holds, a new run
    starts from a random mean with step size ``sigma0``, and the population size is
    multiplied by ``incpopsize``. The number of parents is always lambda // 2. The best
    solution over all runs is the result.

    Parameters
    ----------
    pop_size : population size (lambda) of the first run. None (default) gives the CMA-ES
        default 4 + floor(3 ln n), where n is the number of codes.
    sigma0 : initial step size of every run, in code units (the search space is [0, 1] per
        code). Default 0.5, half the width of the search space, as in the paper.
    x0 : initial mean of the first run as a code vector in [0, 1]. None (default) draws it at
        random. Later runs always start from a random mean.
    incpopsize : factor that multiplies lambda at each restart. Default 2.
    max_restarts : maximum number of restarts. None (default) means no limit. When the limit
        is reached and a restart criterion holds, the run stops with the reason "max_restarts".
    tol_fun : restart when the range of the best values of the last 10 + ceil(30 n / lambda)
        iterations and all values of the current iteration is below this. Default 1e-12.
        Use 0 to turn it off (it stops a run that has already converged to an exact optimum).
    tol_hist_fun : restart when 10 + ceil(30 n / lambda) iterations have passed and the range
        of their best values is below this. Equal values, also infinite ones, have range 0.
        Default 1e-12. Use 0 to turn it off.
    tol_x : restart when sigma times max(sqrt(C_ii), |pc_i|) is below this for every
        coordinate i. None (default) gives 1e-12 * sigma0.
    max_condition : restart when the condition number of C exceeds this. Default 1e14.

    Two more criteria are always on: NoEffectAxis (adding 0.1 standard deviation along a
    principal axis, one axis per iteration in turn, does not change the mean) and
    NoEffectCoord (adding 0.2 standard deviation in some coordinate does not change it).
    The criteria based on objective values use only iterations whose samples are all
    feasible, and are measured in "smaller is better" units, so they work for maximization.

    After a run, ``restarts`` is the number of restarts, ``lam_current`` the population size
    of the last run, and ``restart_log`` a list of (nfe, new lambda, reason) for each restart.

    References: A. Auger and N. Hansen, "A restart CMA evolution strategy with increasing
    population size", IEEE Congress on Evolutionary Computation (CEC 2005), pp. 1769 to 1776.
    N. Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772 (2016).
    """

    name = "CMA-ES with Restarts and Increasing Population Size"
    short_name = "IPOP-CMA-ES"

    mu = None  # always lambda // 2, not a setting (kept as a class attribute for pickling)

    def __init__(self, *, pop_size=None, sigma0=0.5, x0=None, incpopsize=2, max_restarts=None,
                 tol_fun=1e-12, tol_hist_fun=1e-12, tol_x=None, max_condition=1e14, **kwargs):
        self.incpopsize = incpopsize
        self.max_restarts = max_restarts
        self.tol_fun = tol_fun
        self.tol_hist_fun = tol_hist_fun
        self.tol_x = tol_x
        self.max_condition = max_condition
        kwargs.pop("mu", None)  # mu is always lambda // 2 here
        super().__init__(pop_size=pop_size, mu=None, sigma0=sigma0, x0=x0, **kwargs)

    def params(self):
        p = super().params()
        p.pop("mu", None)
        return p

    @property
    def lam(self) -> int:
        """Population size of the current run (lambda)."""
        return self.lam_current

    def validate(self):
        super().validate()
        check(is_real(self.incpopsize) and self.incpopsize > 1, "incpopsize must be greater than 1")
        check(self.max_restarts is None or is_int(self.max_restarts, 0),
              "max_restarts must be a nonnegative integer or None")
        for name in ("tol_fun", "tol_hist_fun"):
            check(is_real(getattr(self, name), 0), f"{name} must be a nonnegative number")
        check(self.tol_x is None or is_real(self.tol_x, 0), "tol_x must be a nonnegative number or None")
        check(is_real(self.max_condition) and self.max_condition > 1, "max_condition must be greater than 1")

    def initialize(self):
        self.lam_current = CMAES.lam.fget(self)
        self.restarts = 0
        self.restart_log = []
        self._start_run()

    def _start_run(self):
        super().initialize()
        self.gen = 0  # iterations of the current run
        self.best_hist = []

    def _restart(self, reason):
        self.restarts += 1
        # grow by at least one, so a factor close to 1 still changes the run
        self.lam_current = max(self.lam_current + 1, int(round(self.lam_current * self.incpopsize)))
        self.restart_log.append((self.nfe, self.lam_current, reason))
        x0, self.x0 = self.x0, None  # only the first run starts at x0
        try:
            self._start_run()
        finally:
            self.x0 = x0

    def result_extra(self):
        return {"restarts": self.restarts,
                "restart_log": [(int(n), int(lam), str(r)) for n, lam, r in self.restart_log]}

    def stop_criterion(self):
        """Name of the restart criterion that holds for the current run, or None."""
        if self.gen == 0:
            return None
        n, sigma = self.dim, self.sigma
        L = 10 + int(np.ceil(30 * n / self.lam_current))
        hist = self.best_hist[-L:]
        # both need a full history, so a flat objective cannot double lambda every iteration
        if len(hist) >= L and np.all(self.pop.V <= 0):
            if _span(np.r_[hist, self.min_keys(self.pop)]) < self.tol_fun:
                return "tol_fun"
        if len(hist) >= L and _span(np.asarray(hist)) < self.tol_hist_fun:
            return "tol_hist_fun"
        tol_x = 1e-12 * self.sigma0 if self.tol_x is None else self.tol_x
        if np.all(sigma * np.maximum(np.sqrt(np.diag(self.C)), np.abs(self.pc)) < tol_x):
            return "tol_x"
        i = self.gen % n
        if np.all(self.mean == self.mean + 0.1 * sigma * self.D[i] * self.B[:, i]):
            return "no_effect_axis"
        if np.any(self.mean == self.mean + 0.2 * sigma * np.sqrt(np.diag(self.C))):
            return "no_effect_coord"
        if (self.D.max() / self.D.min()) ** 2 > self.max_condition:
            return "condition_cov"
        return None

    def iterate(self):
        reason = self.stop_criterion()
        if reason:
            if self.max_restarts is not None and self.restarts >= self.max_restarts:
                self.request_stop("max_restarts")
                return
            self._restart(reason)
        self.gen += 1
        it, self.iter = self.iter, self.gen  # CMAES uses self.iter as the run's iteration count
        try:
            super().iterate()
        finally:
            self.iter = it
        if np.all(self.pop.V <= 0):
            self.best_hist.append(float(self.min_keys(self.pop).min()))
        else:
            self.best_hist = []
