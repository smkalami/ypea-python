"""Parameter spaces: the tunable settings of each algorithm, with their ranges.

Each parameter has a type (real, integer or categorical), a typical range for tuning
(``low`` to ``high``, optionally on a log scale) and the valid range that the algorithm's
``validate()`` accepts. Tuning samples only from the typical range. A ``Space`` also lists
the constructor settings that are left out on purpose, each with the reason.

::

    from ypea.tuning import Space, categorical, integer, real, space

    space(ypea.DE)                       # the built in space of DE
    my = Space({"crossover_prob": real(0.5, 1.0), "pop_size": integer(20, 100)})

``pop_size`` is tuned like any other setting, except for algorithms that choose it from
the dimension (CMA-ES, IPOP-CMA-ES, L-SHADE, Nelder-Mead). The other common settings
(``max_iter``, ``max_nfe``, ``target``, ``max_time``, ``display``) are never tuned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import algorithms as alg

#: settings of the base class that are not search parameters
COMMON = ("max_iter", "pop_size", "max_nfe", "target", "max_time", "display")


@dataclass(frozen=True)
class Param:
    """One tunable setting. Build it with ``real``, ``integer`` or ``categorical``.

    kind : "real", "int" or "cat"
    low, high : typical range for tuning (both included)
    log : sample on a log scale (needs low > 0)
    values : the choices of a categorical setting
    valid : (lowest, highest) value that the algorithm accepts, None for no limit. For
        documentation and checks, tuning never leaves [low, high].
    """

    kind: str
    low: float | None = None
    high: float | None = None
    log: bool = False
    values: tuple = ()
    valid: tuple = (None, None)

    def __post_init__(self):
        if self.kind == "cat":
            if not self.values:
                raise ValueError("a categorical setting needs at least one value")
            return
        if self.kind not in ("real", "int") or not self.low <= self.high:
            raise ValueError(f"need kind 'real' or 'int' and low <= high, got {self}")
        if self.log and self.low <= 0:
            raise ValueError("a log scale needs low > 0")

    def _span(self):
        top = self.high + 1 if self.kind == "int" else self.high
        return (math.log(self.low), math.log(top)) if self.log else (self.low, top)

    def from_unit(self, u):
        """The value at position u in [0, 1] of the range (plain Python types)."""
        a, b = self._span()
        x = a + u * (b - a)
        x = math.exp(x) if self.log else x
        if self.kind == "int":
            return min(int(math.floor(x)), int(self.high))
        return min(max(float(f"{x:.4g}"), self.low), self.high)  # 4 significant digits read well in papers

    def to_unit(self, value):
        """The inverse of ``from_unit`` (an integer maps to the middle of its cell)."""
        a, b = self._span()
        x = value + 0.5 if self.kind == "int" else value
        x = math.log(x) if self.log else x
        return min(max((x - a) / (b - a), 0.0), 1.0) if b > a else 0.5


def real(low, high, log=False, valid=(None, None)) -> Param:
    return Param("real", float(low), float(high), log, valid=valid)


def integer(low, high, log=False, valid=(None, None)) -> Param:
    return Param("int", int(low), int(high), log, valid=valid)


def categorical(*values) -> Param:
    return Param("cat", values=tuple(values))


@dataclass(frozen=True)
class Space:
    """Tunable settings (``params``, name to ``Param``) and the constructor settings left
    out on purpose (``fixed``, name to the reason)."""

    params: dict
    fixed: dict = field(default_factory=dict)

    def sample(self, rng) -> dict:
        """One setting drawn uniformly from the typical ranges."""
        out = {}
        for name, p in self.params.items():
            out[name] = p.values[int(rng.integers(len(p.values)))] if p.kind == "cat" else p.from_unit(rng.random())
        return out

    def neighbor(self, rng, parent, sigma, probs) -> dict:
        """A setting near ``parent``: numbers from a normal distribution with standard
        deviation ``sigma`` (in units of the range) truncated to the range, categories from
        ``probs`` (name to a list of probabilities, one per value)."""
        out = {}
        for name, p in self.params.items():
            if p.kind == "cat":
                out[name] = p.values[int(rng.choice(len(p.values), p=probs[name]))]
                continue
            mean = p.to_unit(parent[name])
            for _ in range(100):  # truncated normal by rejection, accepts about half the draws at worst
                u = mean + sigma * float(rng.standard_normal())
                if 0.0 <= u <= 1.0:
                    break
            out[name] = p.from_unit(min(max(u, 0.0), 1.0))
        return out


def _common(pop=None, pop_reason=None):
    if pop is not None:
        return {"pop_size": pop}, {}
    return {}, {"pop_size": pop_reason}


_AUTO_POP = ("None lets the algorithm choose it from the dimension, so a tuned number would not carry over "
             "to other dimensions")
_DAMP = real(0.9, 1.0, valid=(0, None))
_STEP = real(0.01, 0.3, log=True, valid=(0, None))
_POS = (0, None)
_UNIT = (0, 1)


def _space(params, pop=None, pop_reason=_AUTO_POP, **fixed):
    p, f = _common(pop, pop_reason)
    return Space({**p, **params}, {**f, **fixed})


_MO_FIXED = {"crossover": "operator objects are chosen by the user, not sampled",
             "mutation": "operator objects are chosen by the user, not sampled",
             "max_archive": "the size of the front is chosen by the user",
             "space": "the choice of search space (codes or native operators) is a modeling decision",
             "operators": "user supplied operator objects cannot be sampled"}


SPACES = {
    alg.ABC: _space({
        "onlooker_count": integer(5, 100, log=True, valid=_POS),
        "max_acceleration": real(0.5, 2.0, valid=_POS),
        "fitness": categorical("auto", "rank"),
    }, integer(10, 100, valid=(2, None)),
        trial_limit="None gives onlooker_count times the dimension (Karaboga 2005), which a fixed number "
                    "cannot match across dimensions"),
    alg.ACOR: _space({
        "sample_count": integer(2, 100, log=True, valid=(1, None)),
        "q": real(1e-4, 1.0, log=True, valid=_POS),
        "zeta": real(0.1, 2.0, valid=_POS),
    }, integer(10, 100, valid=(2, None))),
    alg.BA: _space({
        "type": categorical("standard", "probabilistic"),
        "selected_site_ratio": real(0.1, 0.9, valid=_UNIT),
        "selected_site_bee_ratio": real(0.02, 0.5, valid=_UNIT),
        "elite_site_ratio": real(0.1, 0.9, valid=_UNIT),
        "elite_site_bee_ratio": real(1.0, 4.0, valid=_POS),
        "recruited_bee_ratio": real(0.02, 0.5, valid=_UNIT),
        "dance_radius": real(0.01, 0.5, log=True, valid=_POS),
        "dance_radius_damp": _DAMP,
        "neighborhood": categorical("patch", "component"),
    }, integer(10, 100, valid=(1, None))),
    alg.BBO: _space({
        "keep_rate": real(0.0, 0.5, valid=_UNIT),
        "alpha": real(0.5, 1.0, valid=_POS),
        "mutation_prob": real(0.0, 0.5, valid=_UNIT),
        "mutation_step_size": _STEP,
        "mutation_step_size_damp": _DAMP,
    }, integer(10, 100, valid=(1, None))),
    alg.CA: _space({
        "accept_rate": real(0.05, 0.5, valid=_UNIT),
        "influence_type": categorical(1, 2, 3, 4),
        "alpha": real(0.1, 1.5, valid=_POS),
        "beta": real(0.5, 2.0, valid=_POS),
        "selection": categorical(True, False),
    }, integer(10, 100, valid=(1, None))),
    alg.CMAES: _space({
        "sigma0": real(0.05, 0.5, log=True, valid=_POS),
    }, mu="None gives pop_size // 2, the standard choice of the tutorial",
        x0="a starting point belongs to the problem, not to the algorithm"),
    alg.DE: _space({
        "base_vector": categorical("rand", "best", "current-to-best", "rand-to-best"),
        "diff_vectors_count": integer(1, 2, valid=(1, None)),
        "crossover_method": categorical("bin", "exp"),
        "crossover_prob": real(0.0, 1.0, valid=_UNIT),
        "beta_min": real(0.1, 0.5, valid=_POS),
        "beta_max": real(0.5, 1.0, valid=_POS),
    }, integer(10, 200, valid=(4, None)),
        strategy="a short form of base_vector, diff_vectors_count and crossover_method, which are tuned directly"),
    alg.FA: _space({
        "gamma": real(0.01, 10.0, log=True, valid=_POS),
        "beta_base": real(0.5, 2.5, valid=_POS),
        "alpha": real(0.01, 1.0, log=True, valid=_POS),
        "alpha_damp": _DAMP,
        "delta": real(0.01, 0.2, log=True, valid=_POS),
        "m": real(1.0, 3.0, valid=(1, None)),
    }, integer(10, 100, valid=(1, None))),
    alg.GA: _space({
        "crossover_prob": real(0.2, 1.0, valid=_POS),
        "crossover_inflation": real(0.0, 0.5, valid=_POS),
        "mutation_prob": real(0.0, 0.5, valid=_POS),
        "mutation_rate": real(0.01, 0.5, valid=_UNIT),
        "mutation_step_size": _STEP,
        "mutation_step_size_damp": _DAMP,
        "selection": categorical("roulettewheel", "tournament", "random"),
        "selection_pressure": real(0.5, 10.0, valid=_POS),
        "tournament_size": integer(2, 8, valid=(1, None)),
    }, integer(10, 200, valid=(1, None)),
        space="the choice of search space (codes or native operators) is a modeling decision",
        operators="user supplied operator objects cannot be sampled",
        crossover="operator objects are chosen by the user, not sampled",
        mutation="operator objects are chosen by the user, not sampled"),
    alg.HS: _space({
        "new_count": integer(1, 50, log=True, valid=(1, None)),
        "hmcr": real(0.7, 0.99, valid=_UNIT),
        "par": real(0.05, 0.5, valid=_UNIT),
        "fret_width": real(0.001, 0.1, log=True, valid=_POS),
        "fret_width_damp": _DAMP,
    }, integer(5, 50, valid=(1, None))),
    alg.ICA: _space({
        "empire_count": integer(2, 15, valid=(2, None)),
        "selection_pressure": real(0.5, 5.0, valid=_POS),
        "assimilation_coeff": real(0.5, 3.0, valid=_POS),
        "revolution_prob": real(0.0, 0.3, valid=_UNIT),
        "revolution_rate": real(0.01, 0.5, valid=_UNIT),
        "revolution_step_size": _STEP,
        "revolution_step_size_damp": _DAMP,
        "zeta": real(0.0, 0.5, valid=_POS),
        "competition": categorical("roulette", "paper"),
    }, integer(30, 200, valid=(3, None))),
    alg.IPOPCMAES: _space({
        "sigma0": real(0.1, 0.6, valid=_POS),
        "incpopsize": real(1.5, 4.0, valid=(1, None)),
    }, x0="a starting point belongs to the problem, not to the algorithm",
        max_restarts="None lets the evaluation budget end the run, as in the paper",
        tol_fun="a restart tolerance of the paper, it decides when to restart, not how to search",
        tol_hist_fun="a restart tolerance of the paper, it decides when to restart, not how to search",
        tol_x="a restart tolerance of the paper, it decides when to restart, not how to search",
        max_condition="a numerical safeguard of the paper, not a search setting"),
    alg.IWO: _space({
        "initial_pop_size": integer(2, 10, valid=(1, None)),
        "min_seed_count": integer(0, 2, valid=(0, None)),
        "max_seed_count": integer(3, 10, valid=(1, None)),
        "step_size": real(0.05, 0.5, log=True, valid=_POS),
        "final_step_size": real(1e-4, 0.01, log=True, valid=_POS),
        "modulation_index": real(1.0, 4.0, valid=_POS),
        "fitness": categorical("auto", "rank"),
    }, integer(10, 50, valid=(1, None))),
    alg.JADE: _space({
        "p_best_rate": real(0.02, 0.3, valid=_UNIT),
        "c": real(0.02, 0.3, valid=_UNIT),
        "archive": categorical(True, False),
    }, integer(20, 200, valid=(4, None))),
    alg.LSHADE: _space({
        "min_pop_size": integer(4, 10, valid=(4, None)),
        "memory_size": integer(2, 20, valid=(1, None)),
        "archive_rate": real(0.5, 3.0, valid=_POS),
        "p_best_rate": real(0.05, 0.3, valid=_UNIT),
    }),
    alg.NelderMead: _space({
        "reflection": real(0.5, 1.5, valid=_POS),
        "expansion": real(1.6, 3.0, valid=(1, None)),
        "contraction": real(0.3, 0.8, valid=_UNIT),
        "shrink": real(0.3, 0.8, valid=_UNIT),
        "initial_step": real(0.02, 0.5, log=True, valid=_UNIT),
    }, pop_reason="not used, the simplex always has d + 1 points",
        restart_tol="a numerical tolerance for restarts, not a search setting"),
    alg.NSGAII: _space({}, integer(20, 200, valid=(1, None)), **_MO_FIXED),
    alg.NSGAIII: _space({}, pop_reason="None gives the paper's choice from the number of reference points",
                        divisions="set by the number of objectives (Table I of the paper)", **_MO_FIXED),
    alg.PSO: _space({
        "w": real(0.3, 1.0, valid=_POS),
        "wdamp": real(0.9, 1.0, valid=_UNIT),
        "c1": real(0.5, 2.5, valid=_POS),
        "c2": real(0.5, 2.5, valid=_POS),
    }, integer(10, 100, valid=(1, None)),
        w_final="an optional schedule that replaces wdamp, off by default",
        max_velocity="an optional limit, off by default"),
    alg.RandomSearch: _space({}, integer(10, 200, valid=(1, None)),
                             max_archive="the size of the front is chosen by the user"),
    alg.SA: _space({
        "max_sub_iter": integer(1, 20, valid=(1, None)),
        "initial_temp": real(1.0, 1e4, log=True, valid=_POS),
        "final_temp": real(1e-4, 1.0, log=True, valid=_POS),
        "move_count": integer(1, 10, valid=(1, None)),
        "mutation_rate": real(0.01, 0.5, valid=_UNIT),
        "mutation_step_size": _STEP,
        "mutation_step_size_damp": _DAMP,
    }, integer(2, 20, valid=(1, None)),
        space="the choice of search space (codes or native operators) is a modeling decision",
        operators="user supplied operator objects cannot be sampled",
        mutation="operator objects are chosen by the user, not sampled"),
    alg.SHADE: _space({
        "memory_size": integer(2, 100, log=True, valid=(1, None)),
        "archive_rate": real(0.0, 2.0, valid=_POS),
        "p_best_max": real(0.1, 0.4, valid=_UNIT),
    }, integer(20, 200, valid=(4, None))),
    alg.TLBO: _space({}, integer(10, 100, valid=(2, None))),
    alg.MOEAD: _space({
        "neighbors": integer(5, 40, valid=(2, None)),
        "decomposition": categorical("tchebycheff", "pbi"),
        "theta": real(1.0, 10.0, valid=_POS),
    }, integer(20, 200, valid=(1, None)),
        crossover="operator objects are chosen by the user, not sampled",
        mutation="operator objects are chosen by the user, not sampled",
        space="the choice of search space (codes or native operators) is a modeling decision",
        operators="user supplied operator objects cannot be sampled",
        batch="evaluating one child at a time or a whole generation is a choice of how to run the objective",
        max_archive="the size of the front is chosen by the user"),
    alg.MOPSO: _space({
        "w": real(0.1, 0.9, valid=_POS),
        "repository_size": integer(50, 200, valid=(1, None)),
        "divisions": integer(5, 50, valid=(1, None)),
        "mutation_rate": real(0.1, 1.0, valid=_POS),
    }, integer(20, 200, valid=(1, None)),
        max_archive="the size of the front is chosen by the user"),
    alg.SPEA2: _space({}, integer(20, 200, valid=(1, None)),
        archive_size="defaults to the population size, as in the paper",
        crossover="operator objects are chosen by the user, not sampled",
        mutation="operator objects are chosen by the user, not sampled",
        space="the choice of search space (codes or native operators) is a modeling decision",
        operators="user supplied operator objects cannot be sampled",
        max_archive="the size of the front is chosen by the user"),
}


def space(algorithm_class) -> Space:
    """The built in parameter space of an algorithm class."""
    try:
        return SPACES[algorithm_class]
    except KeyError:
        raise KeyError(f"no built in space for {algorithm_class.__name__}, pass space=Space({{...}})") from None


def as_space(algorithm_class, given) -> Space:
    """``given`` as a Space: None gives the built in one, a dict of Params is wrapped."""
    if given is None:
        return space(algorithm_class)
    if isinstance(given, Space):
        return given
    if isinstance(given, dict) and all(isinstance(p, Param) for p in given.values()):
        return Space(dict(given))
    raise TypeError("space must be a Space, a dict of name to Param, or None")
