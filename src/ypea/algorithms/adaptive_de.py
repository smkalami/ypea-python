"""Adaptive Differential Evolution: JADE, SHADE and L-SHADE.

All three use the DE/current-to-pbest/1/bin strategy with an external archive and adapt
the scale factor F and the crossover rate CR from the values that produced better trial
vectors. They share one core (``_AdaptiveDE``) and differ in how F and CR are sampled,
how the adaptation memory is updated, and how the population size changes.

Rules common to the three classes:

* A trial vector replaces its parent when it is at least as good (``no_worse``). It counts
  as a success (its F and CR are stored, and the parent goes to the archive) only when it
  is strictly better.
* A trial component outside [0, 1] is set to the midpoint between the parent's component
  and the violated bound, as in the papers.
* The pbest vector is drawn from the best ``max(2, round(p N))`` members (at most N), as in
  the reference code of SHADE and L-SHADE.
* Rows cut by the evaluation budget (infinite violation) never count as successes and never
  enter the archive.
* Weights for the weighted means (SHADE, L-SHADE) are |f_parent - f_trial| on the "smaller
  is better" objective when both are feasible and finite, and |v_parent - v_trial| on the
  constraint violation otherwise. When the weights sum to zero or are not finite, all
  successes get equal weights.
* When no trial vector succeeds in an iteration, the memory is left unchanged.
"""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real


class _AdaptiveDE(Algorithm):
    """Shared core: DE/current-to-pbest/1/bin with an archive. Subclasses define the hooks."""

    def validate(self):
        super().validate()
        check(self.pop_size is None or self.pop_size >= 4, "pop_size must be at least 4")

    # hooks

    def _initial_size(self):
        return self.pop_size

    def _init_memory(self):
        raise NotImplementedError

    def _sample(self, n):
        """F and CR for n trial vectors."""
        raise NotImplementedError

    def _p_rates(self, n):
        raise NotImplementedError

    def _update_memory(self, s_f, s_cr, w):
        raise NotImplementedError

    def _archive_size(self, n):
        raise NotImplementedError

    def _resize(self):
        """Change the population size after selection (L-SHADE)."""

    # helpers

    def _cauchy_f(self, loc):
        """F ~ Cauchy(loc, 0.1), drawn again while F <= 0, and truncated to 1."""
        F = loc + 0.1 * self.rng.standard_cauchy(len(loc))
        bad = F <= 0
        while bad.any():
            F[bad] = loc[bad] + 0.1 * self.rng.standard_cauchy(int(bad.sum()))
            bad = F <= 0
        return np.minimum(F, 1.0)

    def _normal_cr(self, loc):
        """CR ~ N(loc, 0.1), clipped to [0, 1]."""
        return np.clip(loc + 0.1 * self.rng.standard_normal(len(loc)), 0.0, 1.0)

    def _weights(self, old, new):
        fo, fn = self.min_keys(old), self.min_keys(new)
        with np.errstate(invalid="ignore", over="ignore"):
            use_f = (old.V <= 0) & (new.V <= 0) & np.isfinite(fo) & np.isfinite(fn)
            d = np.where(use_f, np.abs(fo - fn), np.abs(old.V - new.V))
            s = d.sum()
        if not (np.isfinite(s) and s > 0):
            return np.full(len(d), 1.0 / len(d))
        return d / s

    def _trim_archive(self, size):
        if len(self._archive) > size:
            keep = self.rng.permutation(len(self._archive))[:size]
            self._archive = self._archive[np.sort(keep)]

    # algorithm

    def initialize(self):
        self.pop = self.random_population(self._initial_size())
        self._archive = np.empty((0, self.dim))
        self._init_memory()

    def iterate(self):
        X = self.pop.X
        n, d = X.shape
        rows = np.arange(n)
        F, CR = self._sample(n)

        top = np.clip(np.round(self._p_rates(n) * n).astype(int), 2, n)
        pbest = self.argsort(self.pop)[(self.rng.random(n) * top).astype(int)]
        r1 = (rows + 1 + self.rng.integers(n - 1, size=n)) % n  # never equal to i
        pool = np.concatenate([X, self._archive])
        r2 = self.rng.integers(len(pool), size=n)
        bad = (r2 == rows) | (r2 == r1)
        while bad.any():
            r2[bad] = self.rng.integers(len(pool), size=int(bad.sum()))
            bad = (r2 == rows) | (r2 == r1)

        f = F[:, None]
        v = X + f * (X[pbest] - X) + f * (X[r1] - pool[r2])
        v = np.where(v < 0, X / 2, v)
        v = np.where(v > 1, (X + 1) / 2, v)
        mask = self.rng.random((n, d)) < CR[:, None]
        mask[rows, self.rng.integers(d, size=n)] = True
        new = self.evaluate(np.where(mask, v, X))

        valid = np.isfinite(new.V)
        success = self.better(new, self.pop) & valid
        if success.any():
            self._archive = np.concatenate([self._archive, X[success]])
            w = self._weights(self.pop[success], new[success])
            self._update_memory(F[success], CR[success], w)
        self.pop.replace(self.no_worse(new, self.pop) & valid, new)
        self._resize()
        self._trim_archive(self._archive_size(len(self.pop)))


class JADE(_AdaptiveDE):
    """JADE: adaptive DE with an optional external archive (Zhang and Sanderson, 2009).

    Strategy DE/current-to-pbest/1/bin. Each trial vector i gets
    CR_i ~ N(mu_CR, 0.1) clipped to [0, 1] and F_i ~ Cauchy(mu_F, 0.1), drawn again while
    F_i <= 0 and truncated to 1. After each iteration, with S_CR and S_F the values of the
    successful trial vectors, mu_CR = (1 - c) mu_CR + c mean(S_CR) and
    mu_F = (1 - c) mu_F + c lehmer_mean(S_F). Both start at 0.5.

    Parameters
    ----------
    pop_size : population size N (default 100, as in the paper for 30 dimensions).
    p_best_rate : p, the fraction of the best members that pbest is drawn from (default 0.05).
    c : learning rate of mu_CR and mu_F (default 0.1).
    archive : keep the external archive of replaced parents, at most N members (default True).

    See the module docstring for the rules shared with SHADE and L-SHADE.

    Reference: J. Zhang and A. C. Sanderson, "JADE: Adaptive differential evolution with
    optional external archive", IEEE Transactions on Evolutionary Computation 13(5),
    945 to 958 (2009).
    """

    name = "Adaptive Differential Evolution with Optional External Archive"
    short_name = "JADE"

    def __init__(self, *, p_best_rate=0.05, c=0.1, archive=True, **kwargs):
        self.p_best_rate = p_best_rate
        self.c = c
        self.archive = archive
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_real(self.p_best_rate, 0, 1) and self.p_best_rate > 0, "p_best_rate must be in (0, 1]")
        check(is_real(self.c, 0, 1), "c must be in [0, 1]")
        check(isinstance(self.archive, bool), "archive must be True or False")

    def _init_memory(self):
        self.mu_f, self.mu_cr = 0.5, 0.5

    def _sample(self, n):
        return self._cauchy_f(np.full(n, self.mu_f)), self._normal_cr(np.full(n, self.mu_cr))

    def _p_rates(self, n):
        return np.full(n, self.p_best_rate)

    def _update_memory(self, s_f, s_cr, w):
        c = self.c
        self.mu_cr = (1 - c) * self.mu_cr + c * s_cr.mean()
        self.mu_f = (1 - c) * self.mu_f + c * np.sum(s_f**2) / np.sum(s_f)

    def _archive_size(self, n):
        return n if self.archive else 0


class _HistoryDE(_AdaptiveDE):
    """Success history memories M_F and M_CR of size H (SHADE and L-SHADE)."""

    #: L-SHADE: weighted Lehmer mean for CR and the terminal value
    lshade_cr = False

    def _init_memory(self):
        self.m_f = np.full(self.memory_size, 0.5)
        self.m_cr = np.full(self.memory_size, 0.5)  # NaN is the terminal value of L-SHADE
        self.k = 0

    def _sample(self, n):
        r = self.rng.integers(self.memory_size, size=n)
        m_cr = self.m_cr[r]
        terminal = np.isnan(m_cr)
        CR = self._normal_cr(np.where(terminal, 0.0, m_cr))
        CR[terminal] = 0.0
        return self._cauchy_f(self.m_f[r]), CR

    def _update_memory(self, s_f, s_cr, w):
        k = self.k
        self.m_f[k] = np.sum(w * s_f**2) / np.sum(w * s_f)
        if not self.lshade_cr:
            self.m_cr[k] = np.sum(w * s_cr)
        elif np.isnan(self.m_cr[k]) or s_cr.max() == 0:
            self.m_cr[k] = np.nan
        elif np.sum(w * s_cr) > 0:  # zero when every success with CR > 0 has weight 0: keep the slot
            self.m_cr[k] = np.sum(w * s_cr**2) / np.sum(w * s_cr)
        self.k = (k + 1) % self.memory_size

    def _archive_size(self, n):
        return int(round(self.archive_rate * n))


class SHADE(_HistoryDE):
    """Success History based Adaptive DE, version 1.0 (Tanabe and Fukunaga, 2013).

    This is the 2013 algorithm: arithmetic mean for CR and no terminal value. The L-SHADE
    paper (2014) calls L-SHADE without population reduction "SHADE 1.1". It differs from
    this class in the CR update (weighted Lehmer mean with a terminal value).

    Strategy DE/current-to-pbest/1/bin with an archive. Each trial vector i picks a random
    memory slot r and draws CR_i ~ N(M_CR[r], 0.1) clipped to [0, 1] and
    F_i ~ Cauchy(M_F[r], 0.1), drawn again while F_i <= 0 and truncated to 1. Its pbest
    rate p_i is uniform in [2/N, p_best_max]. After each iteration with at least one
    success, slot k (cycling through the memory) gets the weighted Lehmer mean of S_F and
    the weighted arithmetic mean of S_CR, with weights proportional to the improvements.
    All slots start at 0.5.

    Parameters
    ----------
    pop_size : population size N (default 100, as in the paper).
    memory_size : H, the number of memory slots (default 100, as in the paper).
    archive_rate : the archive holds at most round(archive_rate * N) members (default 1).
    p_best_max : upper limit of p_i (default 0.2).

    See the module docstring for the rules shared with JADE and L-SHADE.

    Reference: R. Tanabe and A. Fukunaga, "Success-history based parameter adaptation for
    differential evolution", IEEE Congress on Evolutionary Computation (CEC 2013), 71 to 78.
    """

    name = "Success History based Adaptive Differential Evolution"
    short_name = "SHADE"

    def __init__(self, *, memory_size=100, archive_rate=1.0, p_best_max=0.2, **kwargs):
        self.memory_size = memory_size
        self.archive_rate = archive_rate
        self.p_best_max = p_best_max
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_int(self.memory_size, 1), "memory_size must be a positive integer")
        check(is_real(self.archive_rate, 0), "archive_rate must be a number >= 0")
        check(is_real(self.p_best_max, 0, 1) and self.p_best_max > 0, "p_best_max must be in (0, 1]")

    def _p_rates(self, n):
        low = 2 / n
        return self.rng.uniform(low, max(low, self.p_best_max), n)


class LSHADE(_HistoryDE):
    """SHADE with Linear Population Size Reduction (Tanabe and Fukunaga, 2014).

    Works like SHADE with three changes from the paper. The pbest rate p is fixed. CR is
    updated with the weighted Lehmer mean, and a slot gets the terminal value (shown as NaN
    in ``m_cr``) when it already has it or when all successful CR values are 0. Trial
    vectors from a terminal slot use CR = 0. If the weighted Lehmer mean of CR is undefined
    (every success with CR > 0 has weight 0), the slot keeps its values and k still
    advances. After each iteration the population size
    becomes round(N_init + (N_min - N_init) * progress), and the worst members are removed.

    The progress is ``nfe / max_nfe`` when ``max_nfe`` is set, as in the paper. Without
    ``max_nfe`` it is the fraction of the planned iterations used (``max_iter``, or 100 when
    the run is limited only by ``max_time`` or not at all), so the population still reaches
    N_min at the end of the run. When ``max_iter`` ends the run before ``max_nfe``, the
    population stops shrinking early.

    Parameters
    ----------
    pop_size : initial population size N_init. None (default) gives 18 D, where D is the
        number of codes.
    min_pop_size : final population size N_min (default 4).
    memory_size : H, the number of memory slots (default 6).
    archive_rate : the archive holds at most round(archive_rate * N) members, where N is
        the current population size (default 2.6).
    p_best_rate : p, the fraction of the best members that pbest is drawn from (default 0.11).

    See the module docstring for the rules shared with JADE and SHADE.

    Reference: R. Tanabe and A. Fukunaga, "Improving the search performance of SHADE using
    linear population size reduction", IEEE Congress on Evolutionary Computation
    (CEC 2014), 1658 to 1665.
    """

    name = "Success History based Adaptive Differential Evolution with Linear Population Size Reduction"
    short_name = "L-SHADE"
    auto_pop_size = True
    lshade_cr = True

    def __init__(self, *, pop_size=None, min_pop_size=4, memory_size=6, archive_rate=2.6,
                 p_best_rate=0.11, **kwargs):
        self.min_pop_size = min_pop_size
        self.memory_size = memory_size
        self.archive_rate = archive_rate
        self.p_best_rate = p_best_rate
        super().__init__(pop_size=pop_size, **kwargs)

    def validate(self):
        super().validate()
        check(is_int(self.min_pop_size, 4), "min_pop_size must be an integer >= 4")
        check(self.pop_size is None or self.pop_size >= self.min_pop_size, "pop_size must be at least min_pop_size")
        check(is_int(self.memory_size, 1), "memory_size must be a positive integer")
        check(is_real(self.archive_rate, 0), "archive_rate must be a number >= 0")
        check(is_real(self.p_best_rate, 0, 1) and self.p_best_rate > 0, "p_best_rate must be in (0, 1]")

    def _initial_size(self):
        self.n_init = self.pop_size if self.pop_size is not None else max(18 * self.dim, self.min_pop_size)
        return self.n_init

    def _p_rates(self, n):
        return np.full(n, self.p_best_rate)

    def planned_size(self):
        """Population size the linear reduction gives at the current point of the run."""
        if self.max_nfe is not None:
            progress = self.nfe / self.max_nfe
        else:
            progress = self.iter / self.planned_iterations(self.n_init)
        n = self.n_init + (self.min_pop_size - self.n_init) * min(progress, 1.0)
        return int(round(n))

    def _resize(self):
        n = self.planned_size()
        if n < len(self.pop):
            self.pop = self.sort(self.pop)[:n]
