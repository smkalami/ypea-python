"""Base class for all algorithms."""

from __future__ import annotations

import inspect
import numbers
import time

import numpy as np

from .evaluation import Evaluator
from .population import Individual, Population
from .problem import Problem
from .random import StudioRng
from .result import History, Result


def check(cond, message):
    if not cond:
        raise ValueError(message)


def is_int(x, low=None):
    return isinstance(x, numbers.Integral) and not isinstance(x, bool) and (low is None or x >= low)


def is_real(x, low=None, high=None):
    ok = isinstance(x, numbers.Real) and not isinstance(x, bool) and np.isfinite(x)
    return ok and (low is None or x >= low) and (high is None or x <= high)


class Algorithm:
    """Base class. Subclasses implement ``initialize()`` and ``iterate()``.

    Common settings
    ---------------
    max_iter : int or None
        Maximum number of iterations. If None (the default), the run is limited only by
        ``max_nfe`` and ``max_time``, and when neither is set, ``max_iter`` is 100.
    pop_size : int
        Population size.
    max_nfe : int or None
        Maximum number of objective function evaluations. Never exceeded.
    target : float or None
        Stop when a feasible solution at least this good is found (the MATLAB ``goal``).
    max_time : float or None
        Stop after this many seconds (checked between iterations).
    display : bool
        Print one line per iteration.

    Inside ``initialize()`` and ``iterate()`` a subclass can use ``self.rng``,
    ``self.problem``, ``self.pop``, ``self.iter``, ``self.nfe``, ``self.best`` and the
    helpers below. The best solution so far is tracked automatically.
    """

    name = "Abstract Algorithm"
    short_name = "ALG"

    def __init__(self, *, max_iter=None, pop_size=100, max_nfe=None, target=None,
                 max_time=None, display=False):
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.max_nfe = max_nfe
        self.target = target
        self.max_time = max_time
        self.display = display
        self.validate()

    #: iterations used when no limit at all is given
    default_max_iter = 100

    #: True for algorithms that choose their own population size when pop_size is None
    auto_pop_size = False

    @property
    def iter_limit(self):
        """The iteration limit in effect: ``max_iter``, or 100 when no limit is set at all."""
        if self.max_iter is None and self.max_nfe is None and self.max_time is None:
            return self.default_max_iter
        return self.max_iter

    def planned_iterations(self, evals_per_iter):
        """Expected number of iterations, for schedules that must end with the run
        (for example the SA temperature). Uses whichever limit binds first."""
        limits = [self.iter_limit] if self.iter_limit is not None else []
        if self.max_nfe is not None:
            limits.append(int(np.ceil(self.max_nfe / max(evals_per_iter, 1))))
        return max(min(limits), 1) if limits else self.default_max_iter

    def progress(self) -> float:
        """Fraction of the run used before the current iteration, from 0 to 1: the largest
        of nfe / max_nfe, (iter - 1) / (iteration limit - 1) and elapsed / max_time. Use it
        for schedules that must end when the run ends, whichever limit stops it."""
        parts = []
        if self.max_nfe is not None:
            parts.append(self.nfe / self.max_nfe)
        if self.iter_limit is not None:
            parts.append((self.iter - 1) / max(self.iter_limit - 1, 1))
        if self.max_time is not None:
            parts.append((time.perf_counter() - self._t0) / self.max_time)
        return float(min(max(parts), 1.0)) if parts else 0.0

    @property
    def full_name(self):
        return f"{self.name} ({self.short_name})"

    def params(self) -> dict:
        """The settings of this algorithm (the arguments of its constructor)."""
        names = []
        for cls in type(self).__mro__:
            if "__init__" in vars(cls):
                names += [p for p in inspect.signature(cls.__init__).parameters
                          if p not in ("self", "args", "kwargs") and p not in names]
        return {n: getattr(self, n) for n in names if hasattr(self, n)}

    def __getstate__(self):
        # settings only: run state may hold a lambda objective or an executor
        names = self.params()
        return {k: v for k, v in self.__dict__.items() if k in names}

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"

    def validate(self):
        """Check the settings. Subclasses extend it and call ``super().validate()``."""
        check(self.max_iter is None or is_int(self.max_iter, 1), "max_iter must be a positive integer or None")
        check(is_int(self.pop_size, 1) or (self.auto_pop_size and self.pop_size is None),
              "pop_size must be a positive integer" + (" or None" if self.auto_pop_size else ""))
        check(self.max_nfe is None or is_int(self.max_nfe, 1), "max_nfe must be a positive integer or None")
        check(self.target is None or is_real(self.target), "target must be a number or None")
        check(self.max_time is None or is_real(self.max_time, 0), "max_time must be a positive number or None")

    # running

    def solve(self, problem: Problem, *, seed=None, callbacks=(), executor=None) -> Result:
        """Solve ``problem`` and return a ``Result``.

        seed : int, numpy Generator, ypea.StudioRng, or None
            The same integer seed gives the same result. (A NumPy Generator is consumed by
            the run.) A ``StudioRng`` draws the same numbers as YPEA Studio, and each run
            starts its stream from the beginning, so the same object gives the same result.
        callbacks : list of callables
            Each is called as ``cb(algorithm)`` after every iteration. A callback that
            returns True stops the run. Objects may also define ``on_start(algorithm)``
            and ``on_end(algorithm, result)``.
        executor : concurrent.futures.Executor, optional
            Evaluate solutions in parallel (not used for vectorized problems).
        """
        if not isinstance(problem, Problem):
            raise TypeError("problem must be a ypea.Problem")
        self.validate()
        self.problem = problem
        self.seed = seed
        # a StudioRng starts a fresh stream at position 0, so the same object repeats the run
        self.rng = StudioRng(seed.seed) if isinstance(seed, StudioRng) else np.random.default_rng(seed)
        self.evaluator = Evaluator(problem, self.max_nfe, self.target, executor)
        self.iter = 0
        self.pop = None
        self._hist = []
        self._stop_request = None
        self._t0 = time.perf_counter()
        stop_reason = None

        self.initialize()
        self._record()
        for cb in callbacks:
            getattr(cb, "on_start", lambda alg: None)(self)

        while True:
            stop_reason = self._stop_reason()
            if stop_reason:
                break
            self.iter += 1
            nfe_before = self.nfe
            self.iterate()
            self._record()
            if self._stop_request:
                stop_reason = self._stop_request
                break
            if self.nfe == nfe_before:
                # settings that create no new solutions (for example GA with pop_size=1)
                stop_reason = "no_evaluations"
                break
            if self.display:
                print(f"Iteration {self.iter}: Best Obj. Value = {self.best.obj_value[0]:.6g}, "
                      f"NFE = {self.nfe}")
            if any([bool(cb(self)) for cb in callbacks]):
                stop_reason = "callback"
                break

        result = self._result(stop_reason)
        for cb in callbacks:
            getattr(cb, "on_end", lambda alg, res: None)(self, result)
        return result

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def request_stop(self, reason: str):
        """Stop the run after the current iteration, with ``reason`` as the stop reason."""
        self._stop_request = reason

    def _stop_reason(self):
        if self._stop_request:
            return self._stop_request
        if self.evaluator.stop_reason:
            return self.evaluator.stop_reason
        if self.iter_limit is not None and self.iter >= self.iter_limit:
            return "max_iter"
        if self.max_time is not None and time.perf_counter() - self._t0 >= self.max_time:
            return "max_time"
        return None

    def _record(self):
        b = self.best
        self._hist.append((self.iter, self.nfe, b.F[0, 0], b.V[0], time.perf_counter() - self._t0))

    def _result(self, stop_reason):
        it, nfe, f, v, t = (np.array(c) for c in zip(*self._hist, strict=True))
        ev = self.evaluator
        history = History(it, nfe, f, v, t, np.array(ev.trace_nfe), np.array(ev.trace_f),
                          np.array(ev.trace_v))
        b = self.best
        best = Individual(b.X[0].copy(), self.problem.decode_one(b.X[0]), float(b.F[0, 0]), float(b.V[0]))
        return Result(best=best, history=history, algorithm=self.short_name, params=self.params(),
                      problem=self.problem.name, seed=self.seed, nfe=self.nfe,
                      iterations=self.iter, run_time=float(t[-1]), stop_reason=stop_reason,
                      optimum=self.problem.optimum, extra=self.result_extra())

    def result_extra(self) -> dict:
        """Algorithm specific information to keep in ``Result.extra`` (override it)."""
        return {}

    # helpers for subclasses

    @property
    def nfe(self) -> int:
        return self.evaluator.nfe

    @property
    def best(self) -> Population:
        """Best solution found so far, as a Population of length 1."""
        return self.evaluator.best

    @property
    def dim(self) -> int:
        return self.problem.code_size

    @property
    def constrained(self) -> bool:
        """True once the objective has returned constraint values (known after initialization)."""
        return self.evaluator.constrained

    def min_keys(self, pop: Population) -> np.ndarray:
        """Objective values turned into "smaller is better" (negated for maximization)."""
        return self.problem.comparator.sign * pop.F[:, 0]

    def evaluate(self, X) -> Population:
        """Evaluate codes (clipped to [0, 1]). Always returns as many rows as ``X``."""
        return self.evaluator(X)

    def random_population(self, n=None) -> Population:
        return self.evaluate(self.rng.random((self.pop_size if n is None else n, self.dim)))

    def better(self, a: Population, b: Population) -> np.ndarray:
        """Element wise: is a strictly better than b? Broadcasts a length 1 population."""
        return self.problem.comparator.better(a.F, a.V, b.F, b.V)

    def no_worse(self, a: Population, b: Population) -> np.ndarray:
        """Element wise: is a at least as good as b?"""
        return ~self.better(b, a)

    def argsort(self, pop: Population) -> np.ndarray:
        """Indices from best to worst (stable)."""
        return self.problem.comparator.argsort(pop.F, pop.V)

    def sort(self, pop: Population) -> Population:
        return pop[self.argsort(pop)]

    def argbest(self, pop: Population) -> int:
        return int(self.argsort(pop)[0])

    def ranks(self, pop: Population) -> np.ndarray:
        """0 for the best solution, ties share a rank."""
        return self.problem.comparator.ranks(pop.F, pop.V)

    def selection_probs(self, pop: Population, pressure=1.0) -> np.ndarray:
        """Selection probabilities from ranks: p is proportional to exp(-pressure * r),
        where r goes from 0 (best) to 1 (worst). Independent of the scale of the objective."""
        r = self.ranks(pop)
        r = r / max(len(pop) - 1, 1)
        p = np.exp(-pressure * r)
        return p / p.sum()
