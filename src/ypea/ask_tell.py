"""Ask and tell: run an algorithm while you evaluate its solutions yourself.

The algorithm runs its normal ``solve`` loop in a worker thread. Each time it needs a
batch of solutions evaluated, the worker hands the batch over and waits for the values.
So every algorithm works unchanged, and a run that is told the true objective values
gives exactly the ``Result`` that ``solve`` gives with the same seed.
"""

from __future__ import annotations

import copy
import queue
import threading
import weakref

from .evaluation import Evaluator, StopRun, parse_values


class _HandoffEvaluator(Evaluator):
    """An Evaluator that sends each batch to the main thread instead of calling the objective."""

    def __init__(self, problem, max_nfe, target, to_main, to_worker, **options):
        super().__init__(problem, max_nfe, target, **options)
        self.to_main, self.to_worker = to_main, to_worker

    def _evaluate(self, X):
        self.to_main.put(("ask", X.copy(), self.problem.decode(X)))
        kind, payload = self.to_worker.get()
        if kind == "stop":
            raise StopRun(payload)
        F, V, constrained = payload
        self.constrained |= constrained
        return F, V


def _work(alg, problem, seed, callbacks, to_main, to_worker, keep_evaluations):
    # holds no reference to the AskTell handle, so an abandoned handle can be collected
    event = ("error", RuntimeError("the run ended without a result"))
    try:
        ev = _HandoffEvaluator(problem, alg.max_nfe, alg.target, to_main, to_worker,
                               **alg._evaluator_options(keep_evaluations))
        event = ("done", alg._run(problem, seed, callbacks, ev))
    except StopRun:
        event = ("empty", None)  # stopped before any solution was evaluated
    except BaseException as e:  # noqa: BLE001 (handed to the main thread, which raises it)
        event = ("error", e)
    finally:
        to_main.put(event)


class AskTell:
    """A run whose solutions you evaluate yourself. Create it with ``Algorithm.ask_tell``.

    ::

        run = ypea.PSO(max_nfe=500).ask_tell(problem, seed=0)
        while not run.done:
            solutions = run.ask()
            run.tell([my_objective(s) for s in solutions])
        result = run.result()

    ``ask()`` returns the next batch of decoded solutions: a list of ``Solution`` objects,
    or for a vectorized problem one ``Solution`` whose attributes have a leading batch
    axis. Their codes are in ``codes``. ``tell(values)`` takes what the objective would
    return: one value ``f``, ``(f, g)`` or ``(f, g, h)`` per solution, or for a vectorized
    problem the arrays ``f``, ``(f, g)`` or ``(f, g, h)`` for the whole batch. With several
    objectives each ``f`` is a list or array of ``n_obj`` values (an array of shape
    (n, n_obj) for a vectorized problem). The batch size changes from call to
    call (some algorithms ask for one solution at a time), and the run never asks for more
    than ``max_nfe`` evaluations.

    Callbacks run in the worker thread, between ``tell()`` and the return of the next
    batch. ``max_time`` counts the time you spend evaluating.

    Attributes
    ----------
    done : bool
        True when the run has ended. ``result()`` then gives its ``Result``.
    codes : numpy.ndarray or None
        Codes of the batch that ``ask()`` returned last, with shape (n, code_size).
    algorithm : Algorithm
        The copy of the algorithm that runs, for example to read ``iter`` or ``pop``
        in between (only while the run waits for values).
    """

    def __init__(self, algorithm, problem, seed=None, callbacks=(), keep_evaluations=False):
        self.algorithm = copy.copy(algorithm)  # settings only, so a live run shares no state
        self.problem = problem
        self.codes = None
        self.done = False
        self._asked = False
        self._result = None
        self._error = None
        self._to_main, self._to_worker = queue.Queue(), queue.Queue()
        self._thread = threading.Thread(
            target=_work, args=(self.algorithm, problem, seed, list(callbacks), self._to_main, self._to_worker,
                                keep_evaluations),
            name="ypea-ask-tell", daemon=True)
        # an abandoned run ends its worker when the handle is collected
        weakref.finalize(self, self._to_worker.put, ("stop", "stopped"))
        self._thread.start()
        self._wait()

    def _wait(self):
        """Wait for the next event of the worker: a batch to evaluate, or the end of the run."""
        kind, *payload = self._to_main.get()
        if kind == "ask":
            self.codes, self._solutions = payload
            return
        self._thread.join()
        self.done = True
        self.codes = self._solutions = None
        if kind == "done":
            self._result = payload[0]
        elif kind == "error":
            self._error = payload[0]
            raise self._error

    def ask(self):
        """The next batch of decoded solutions to evaluate (their codes are in ``codes``)."""
        if self.done:
            raise RuntimeError("the run is done: call result()")
        if self._asked:
            raise RuntimeError("tell() the values of the last batch before asking again")
        self._asked = True
        return self._solutions

    def tell(self, values):
        """Give the objective values of the batch that ``ask()`` returned, in the same order.

        The run then continues until it asks for the next batch or ends. An exception of
        the algorithm or of a callback is raised here.
        """
        if self.done:
            raise RuntimeError("the run is done: call result()")
        if not self._asked:
            raise RuntimeError("call ask() before tell()")
        # parsed here, so wrong values raise in this thread and the run can go on
        p = self.problem
        parsed = parse_values(values, len(self.codes), p.vectorized, p.n_obj, p.equality_tolerance)
        self._asked = False
        self._to_worker.put(("values", parsed))
        self._wait()

    def stop(self, reason="stopped"):
        """End the run now and return its ``Result``, with ``reason`` as the stop reason."""
        if not self.done:
            self._asked = False
            self._to_worker.put(("stop", reason))
            self._wait()
        return self.result()

    def result(self):
        """The ``Result`` of the run, once it is done."""
        if not self.done:
            raise RuntimeError("the run is not done: keep asking and telling, or call stop()")
        if self._error is not None:
            raise RuntimeError("the run failed") from self._error
        if self._result is None:
            raise RuntimeError("the run was stopped before any solution was evaluated, so it has no result")
        return self._result
