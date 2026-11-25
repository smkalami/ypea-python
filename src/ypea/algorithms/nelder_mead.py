"""Nelder-Mead downhill simplex method."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_real


class NelderMead(Algorithm):
    """Nelder-Mead simplex method with restarts (Nelder and Mead 1965, Lagarias et al. 1998).

    Works on the codes in [0, 1] like every other method. The simplex has d + 1 vertices,
    where d is the number of codes, so ``pop_size`` is not used.

    Start: one random point x0 = rng.random(d). Vertex i (i = 1 to d) is x0 with coordinate
    i - 1 moved by ``initial_step``, downward when the move would pass 1. The d + 1 points
    are evaluated as one batch.

    One iteration (Lagarias et al. 1998, with the comparator in place of "<"): sort the
    simplex from best to worst (stable). Let x1 be the best vertex, xn the second worst, x
    the worst, and c the centroid of all vertices but the worst. Then:

    1. Reflect: r = c + reflection * (c - x).
    2. If r is better than x1, expand: e = c + expansion * (r - c). Keep e if it is better
       than r, otherwise keep r.
    3. Else if r is better than xn, keep r.
    4. Else if r is better than x, contract outside: o = c + contraction * (r - c). Keep o
       if it is at least as good as r, otherwise shrink.
    5. Else contract inside: i = c - contraction * (c - x). Keep i if it is better than x,
       otherwise shrink.
    6. Shrink: every vertex but x1 moves to x1 + shrink * (v - x1). The d new points are
       evaluated as one batch.

    Every trial point is evaluated through ``evaluate``, which clips it to [0, 1], and the
    clipped point is what enters the simplex. All comparisons use the problem's comparison
    rules, so maximization and constraints need no extra code.

    Restart: at the start of an iteration, when the largest distance from the best vertex
    to any other vertex is below ``restart_tol``, the method starts again from a new random
    point, as in "Start". The framework keeps the best solution across restarts, so the
    method can use its whole budget. The only random draws are ``rng.random(d)`` at the
    start and at each restart.

    Parameters
    ----------
    reflection, expansion, contraction, shrink : the coefficients rho, chi, gamma and
        sigma of Lagarias et al. (default 1, 2, 0.5, 0.5)
    initial_step : size of the initial simplex in code units (default 0.1)
    restart_tol : simplex size below which the method restarts (default 1e-10)

    References: J. A. Nelder and R. Mead, "A simplex method for function minimization",
    The Computer Journal 7(4), 1965, 308 to 313. J. C. Lagarias, J. A. Reeds, M. H. Wright
    and P. E. Wright, "Convergence properties of the Nelder-Mead simplex method in low
    dimensions", SIAM Journal on Optimization 9(1), 1998, 112 to 147.
    """

    name = "Nelder-Mead Simplex"
    short_name = "NM"
    auto_pop_size = True

    def __init__(self, *, reflection=1.0, expansion=2.0, contraction=0.5, shrink=0.5,
                 initial_step=0.1, restart_tol=1e-10, pop_size=None, **kwargs):
        self.reflection = reflection
        self.expansion = expansion
        self.contraction = contraction
        self.shrink = shrink
        self.initial_step = initial_step
        self.restart_tol = restart_tol
        super().__init__(pop_size=pop_size, **kwargs)

    def validate(self):
        super().validate()
        check(is_real(self.reflection, 0) and self.reflection > 0, "reflection must be positive")
        check(is_real(self.expansion, 0) and self.expansion > 1 and self.expansion > self.reflection,
              "expansion must be greater than 1 and than reflection")
        check(is_real(self.contraction, 0, 1) and 0 < self.contraction < 1, "contraction must be in (0, 1)")
        check(is_real(self.shrink, 0, 1) and 0 < self.shrink < 1, "shrink must be in (0, 1)")
        check(is_real(self.initial_step, 0, 1) and self.initial_step > 0, "initial_step must be in (0, 1]")
        check(is_real(self.restart_tol, 0), "restart_tol must be non negative")

    def initialize(self):
        self.restarts = 0
        self._start()

    def _start(self):
        x0 = self.rng.random(self.dim)
        V = np.repeat(x0[None, :], self.dim + 1, axis=0)
        i = np.arange(self.dim)
        step = np.where(x0 + self.initial_step > 1, -self.initial_step, self.initial_step)
        V[i + 1, i] += step
        self.pop = self.evaluate(V)

    def iterate(self):
        pop = self.sort(self.pop)
        best = pop[0]
        if np.max(np.linalg.norm(pop.X[1:] - pop.X[0], axis=1)) < self.restart_tol:
            self.restarts += 1
            self._start()
            return
        worst, second = pop[-1], pop[-2]
        c = pop.X[:-1].mean(axis=0)
        x = worst.X[0]

        r = self.evaluate(c + self.reflection * (c - x))
        if self.better(r, best)[0]:
            e = self.evaluate(c + self.expansion * (r.X[0] - c))
            new = e if self.better(e, r)[0] else r
        elif self.better(r, second)[0]:
            new = r
        elif self.better(r, worst)[0]:
            o = self.evaluate(c + self.contraction * (r.X[0] - c))
            new = o if self.no_worse(o, r)[0] else None
        else:
            i = self.evaluate(c - self.contraction * (c - x))
            new = i if self.better(i, worst)[0] else None

        if new is None:  # shrink toward the best vertex
            moved = self.evaluate(best.X + self.shrink * (pop.X[1:] - best.X))
            self.pop = best.concat(best, moved)
        else:
            self.pop = pop.concat(pop[:-1], new)
