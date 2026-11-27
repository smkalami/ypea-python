"""Cultural Algorithm (CA)."""

from __future__ import annotations

import numpy as np

from ..algorithm import Algorithm, check, is_int, is_real


class CA(Algorithm):
    """Cultural Algorithm (Reynolds, 1994).

    A population space and a belief space (the culture) evolve together. In every
    iteration, the best ``accept_rate`` share of the population updates the culture,
    and the culture then moves every individual.

    The belief space has two parts:

    * situational knowledge: the best accepted solution found so far,
    * normative knowledge: for every component j, an interval [xmin_j, xmax_j] with the
      scores L_j and U_j of the solutions that set its ends. An accepted solution moves
      the lower end when its component is below xmin_j or when it is better than L_j
      (and the same for the upper end). Scores are compared with the problem's
      comparator, so maximization and constraints work.

    Parameters
    ----------
    accept_rate : share of the population (sorted from best) that updates the culture.
        At least two solutions are accepted (one would give zero width intervals).
    influence_type : how the culture moves an individual (component j, N is a standard
        normal number, size_j = xmax_j - xmin_j, s is the situational solution):

        1. normative size only: x_j += alpha * size_j * N
        2. situational direction only: x_j moves toward s_j by abs(0.1 * alpha * N)
        3. (default) both: x_j moves toward s_j by abs(alpha * size_j * N)
        4. normative size and direction: x_j moves up by abs(alpha * size_j * N) when it is
           below xmin_j, down when it is above xmax_j, and by beta * alpha * size_j * N
           inside the interval

        When x_j equals s_j (types 2 and 3) the step keeps its random sign.

        The four types follow Reynolds and Chung (1997), with every step multiplied by
        ``alpha`` (``alpha=1`` reproduces types 1, 3 and 4), and type 2 uses the fixed
        step 0.1 * alpha in place of a self adapted strategy parameter.
    alpha : step size
    beta : scaling factor of the step inside the normative interval (type 4)
    selection : True (default) selects the best ``pop_size`` of parents and offspring.
        False replaces every individual by its offspring, as in YPEA for MATLAB (the
        default influence type then does not converge).

    Every individual makes one offspring per iteration (``pop_size`` evaluations). The
    parents and the offspring are merged and the best ``pop_size`` survive. The culture
    is then updated from the survivors.

    Differences: the four influence types follow YPEA for MATLAB (they match the CAEP
    variants of Chung and Reynolds). YPEA for MATLAB replaced every individual by its
    moved copy, without selection. Here the population is selected from parents and
    offspring, as in the evolutionary programming base of CAEP. The survivors
    are chosen by truncation, not by the tournament of evolutionary programming.

    Reference: R. G. Reynolds, "An introduction to cultural algorithms", Proceedings of
    the Third Annual Conference on Evolutionary Programming, 1994, pp. 131 to 139.
    C.-J. Chung and R. G. Reynolds, "CAEP: an evolution-based tool for real-valued
    function optimization using cultural algorithms", International Journal on
    Artificial Intelligence Tools, 7(3), 1998, pp. 239 to 291.
    R. G. Reynolds and C.-J. Chung, "Knowledge-based self-adaptation in evolutionary
    programming using cultural algorithms", Proceedings of the IEEE International
    Conference on Evolutionary Computation, 1997 (the source of the influence types).
    """

    name = "Cultural Algorithm"
    short_name = "CA"

    def __init__(self, *, accept_rate=0.2, influence_type=3, alpha=0.9, beta=1.0, selection=True, **kwargs):
        self.accept_rate = accept_rate
        self.selection = selection
        self.influence_type = influence_type
        self.alpha = alpha
        self.beta = beta
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        check(is_real(self.accept_rate, 0, 1) and self.accept_rate > 0, "accept_rate must be in (0, 1]")
        check(is_int(self.influence_type) and 1 <= self.influence_type <= 4, "influence_type must be 1, 2, 3 or 4")
        check(is_real(self.alpha, 0), "alpha must be non negative")
        check(is_real(self.beta, 0), "beta must be non negative")
        check(isinstance(self.selection, bool), "selection must be True or False")

    def initialize(self):
        # MATLAB round (half away from zero), at least one
        # at least two, or the normative intervals have zero width and the search stalls
        self._accept_count = min(max(2, int(self.accept_rate * self.pop_size + 0.5)), self.pop_size)
        self.pop = self.sort(self.random_population())
        d = self.dim
        worst = self.problem.comparator.worst_value
        self.situational = self.pop[0]
        self.xmin, self.xmax = np.full(d, np.inf), np.full(d, -np.inf)
        self.LF, self.LV = np.full((d, 1), worst), np.full(d, np.inf)
        self.UF, self.UV = np.full((d, 1), worst), np.full(d, np.inf)
        self.adjust_culture()

    def adjust_culture(self):
        """Update the belief space with the accepted (best) part of the sorted population."""
        cmp = self.problem.comparator
        for i in range(min(self._accept_count, len(self.pop))):
            p = self.pop[i]
            if self.better(p, self.situational)[0]:
                self.situational = p
            x = p.X[0]
            low = (x < self.xmin) | cmp.better(p.F, p.V, self.LF, self.LV)
            self.xmin[low], self.LF[low], self.LV[low] = x[low], p.F[0], p.V[0]
            up = (x > self.xmax) | cmp.better(p.F, p.V, self.UF, self.UV)
            self.xmax[up], self.UF[up], self.UV[up] = x[up], p.F[0], p.V[0]

    def iterate(self):
        X = self.pop.X
        size = self.xmax - self.xmin
        t = self.influence_type
        dx = self.alpha * (0.1 if t == 2 else size) * self.rng.standard_normal(X.shape)
        if t in (2, 3):
            s = self.situational.X
            # toward s_j, the step keeps its random sign when x_j == s_j
            dx = np.where(X < s, np.abs(dx), np.where(X > s, -np.abs(dx), dx))
        elif t == 4:
            dx = np.where(X < self.xmin, np.abs(dx), np.where(X > self.xmax, -np.abs(dx), self.beta * dx))
        # selection from parents and offspring (as in CAEP)
        offspring = self.evaluate(X + dx)
        merged = self.pop.concat(self.pop, offspring) if self.selection else offspring
        self.pop = self.sort(merged)[: self.pop_size]
        self.adjust_culture()
