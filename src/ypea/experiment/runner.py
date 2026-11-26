"""The experiment runner: several algorithms on several problems, many runs each."""

from __future__ import annotations

import itertools
import platform
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import indicators
from ..algorithm import Algorithm
from ..random import StudioRng, derive_seed
from ..result import _plain
from . import store

#: fractions of max_nfe at which the CEC 2017 rules record the error
CEC_CHECKPOINTS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class Experiment:
    """Run every algorithm on every problem ``runs`` times, save each run, and report.

    Parameters
    ----------
    algorithms : dict or list
        A dict from label to ``Algorithm``, or a list of algorithms labeled by their
        ``short_name`` (repeated names get " (2)", " (3)" and so on).
    problems : list of Problem
        Problem names must be unique. With ``n_jobs > 1`` the problems must be picklable
        (module level objective functions, no lambdas).
    runs : int
        Number of runs of each algorithm on each problem.
    max_nfe, max_iter : int or None
        The budget of every run. If either is given, both replace the limits of each
        algorithm (a copy is made, the caller's objects are not changed). If both are
        None, each algorithm keeps its own limits.
    seed : int
        Base seed.
    seeds : "numpy" or "studio"
        How run ``i`` gets its random numbers. With "numpy", run ``i`` uses
        ``numpy.random.SeedSequence(seed, spawn_key=(a, p, i))``, where ``a`` and ``p`` are
        stable 32-bit hashes of the label and the problem name: every run has its own
        independent stream. With "studio", run ``i`` uses
        ``ypea.StudioRng(ypea.derive_seed(seed, i))``, as YPEA Studio does, so run ``i``
        gets the same seed for every algorithm and problem (common random numbers). With
        either rule a result does not depend on ``n_jobs``, on the order of the algorithms
        or problems, or on which of them are included.
    n_jobs : int
        Number of worker processes. Results are the same for any value.
    output : path or None
        Folder for the run files and ``metadata.json``. With None, results stay in memory.
    checkpoints : sequence of float or None
        Fractions of ``max_nfe`` at which the best objective value (and the error, when
        the problem has an ``optimum``) is recorded. None gives the CEC 2017 rule
        (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, ..., 1.0). Runs without an evaluation budget
        record no checkpoints, only their final values. With several objectives the
        checkpoints hold IGD and hypervolume instead (see below).

    **Several objectives.** Each run also stores the IGD and hypervolume of its final front,
    and of its front at each checkpoint (rebuilt with ``History.front_at``). Only feasible
    members with finite values count, each objective turned into minimization. A problem
    gives its true front as ``info["pareto_front"]`` (``ypea.benchmarks.multi_problem`` does
    this) or a reference point as ``info["ref_point"]``, both in its own objective values.

    * IGD: ``ypea.indicators.igd`` of the front against ``info["pareto_front"]``, on the raw
      objective values. Problems without a true front have no IGD.
    * Hypervolume: each objective is normalized by the ideal point z and the nadir point n
      of the true front, f' = (f - z) / (n - z), and the reference point is 1.1 in every
      normalized objective, unless ``info["ref_point"]`` gives another one. The best value
      is then a little below 1.1 ** m. Without a true front, the raw objective values and
      ``info["ref_point"]`` are used. Exact for up to 3 objectives, a seeded Monte Carlo
      estimate above (``ypea.indicators.hypervolume``).

    A multi-objective problem with neither ``pareto_front`` nor ``ref_point`` raises an error.

    Each run file holds ``Result.to_dict()``, the checkpoint values, the problem sense and
    a run key. The key is a hash of the algorithm class and settings, the problem name,
    the budget, the seed rule and the checkpoints. ``run()`` skips runs whose file has
    the same key, so an interrupted experiment resumes, and a changed setting is run again.
    Apart from the measured times (``run_time`` and ``history.time``), the files do not
    depend on ``n_jobs``.
    """

    def __init__(self, algorithms, problems, runs=30, max_nfe=None, max_iter=None, seed=0,
                 seeds="numpy", n_jobs=1, output=None, checkpoints=None):
        if not isinstance(algorithms, dict):
            algorithms = _label(algorithms)
        if not algorithms or not all(isinstance(a, Algorithm) for a in algorithms.values()):
            raise TypeError("algorithms must be a non empty dict or list of ypea Algorithms")
        self.algorithms = {str(k): v for k, v in algorithms.items()}
        self.problems = list(problems)
        names = [p.name for p in self.problems]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"problem names must be unique, got {names}")
        for what, values in (("labels", self.algorithms), ("problem names", names)):
            folders = [store.safe_name(v).lower() for v in values]
            if len(set(folders)) != len(folders):
                raise ValueError(f"{what} must stay unique as folder names, got {list(values)}")
        if seeds not in ("numpy", "studio"):
            raise ValueError("seeds must be 'numpy' or 'studio'")
        if seeds == "studio" and not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("with seeds='studio' the seed must be from 0 to 4294967295")
        if not (isinstance(runs, int) and runs >= 1 and isinstance(n_jobs, int) and n_jobs >= 1):
            raise ValueError("runs and n_jobs must be positive integers")
        self.runs, self.max_nfe, self.max_iter = runs, max_nfe, max_iter
        self.seed, self.seeds, self.n_jobs, self.output = int(seed), seeds, n_jobs, output
        self.checkpoints = tuple(CEC_CHECKPOINTS if checkpoints is None else checkpoints)
        self._records = {}
        self._indicators = {p.name: indicator_setup(p) for p in self.problems}
        self._indicator_info = {k: v and _indicator_info(v) for k, v in self._indicators.items()}
        # copies with the budget of the experiment (this also validates the budget)
        self._algs = {k: self._with_budget(a) for k, a in self.algorithms.items()}

    @classmethod
    def sweep(cls, algorithm_class, settings, problems, **kwargs):
        """An experiment over several settings of one algorithm, for tuning studies.

        ``settings`` is a dict from parameter name to a list of values (every combination
        is run), or a list of dicts. Labels name the settings that vary, for example
        ``"DE(F=0.5, CR=0.9)"``. Other keyword arguments go to ``Experiment``.
        """
        if isinstance(settings, dict):
            keys = list(settings)
            combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*settings.values())]
            shown = [k for k in keys if len(settings[k]) > 1]
        else:
            combos = [dict(s) for s in settings]
            shown = None
        algorithms = {}
        for s in combos:
            parts = ", ".join(f"{k}={v}" for k, v in s.items() if shown is None or k in shown)
            algorithms[f"{algorithm_class.short_name}({parts})"] = algorithm_class(**s)
        if len(algorithms) != len(combos):
            raise ValueError("two settings give the same label")
        return cls(algorithms, problems, **kwargs)

    def _with_budget(self, alg):
        budget = {} if self.max_nfe is None and self.max_iter is None else {
            "max_nfe": self.max_nfe, "max_iter": self.max_iter}
        return type(alg)(**{**alg.params(), **budget})

    def _seed(self, label, problem, i):
        """The seed of one run and a plain description of it (part of the run key)."""
        if self.seeds == "studio":
            s = derive_seed(self.seed, i)
            return StudioRng(s), {"studio": s}
        key = (store.name_hash(label), store.name_hash(problem), i)
        return np.random.SeedSequence(self.seed, spawn_key=key), {"entropy": self.seed, "spawn_key": list(key)}

    def _tasks(self):
        for label, alg in self._algs.items():
            nfe = None if alg.max_nfe is None else [max(1, round(c * alg.max_nfe)) for c in self.checkpoints]
            for p in self.problems:
                for i in range(self.runs):
                    seed, seed_info = self._seed(label, p.name, i)
                    settings = {
                        "algorithm": f"{type(alg).__module__}.{type(alg).__qualname__}",
                        "params": _plain(alg.params()), "problem": p.name,
                        "max_nfe": self.max_nfe, "max_iter": self.max_iter,
                        "seeds": self.seeds, "seed": seed_info, "run": i, "checkpoints": nfe,
                    }
                    ind = self._indicators[p.name]
                    if ind is not None:  # several objectives only, so single objective keys stay the same
                        settings["indicators"] = self._indicator_info[p.name]
                    yield {"label": label, "algorithm": alg, "problem": p, "run": i,
                           "seed": seed, "checkpoints": nfe, "settings": settings, "indicators": ind,
                           "key": store.key_hash(settings),
                           "path": None if self.output is None else store.run_path(self.output, label, p.name, i)}

    def metadata(self) -> dict:
        from .. import __version__

        keys = {}
        for t in self._tasks():
            keys.setdefault(t["label"], {}).setdefault(t["problem"].name, []).append(t["key"])
        return _plain({
            "versions": {"ypea": __version__, "numpy": np.__version__, "python": platform.python_version()},
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "started": datetime.now().astimezone().isoformat(timespec="seconds"),
            "runs": self.runs, "max_nfe": self.max_nfe, "max_iter": self.max_iter,
            "seed": self.seed, "seeds": self.seeds, "n_jobs": self.n_jobs,
            "checkpoints": list(self.checkpoints),
            "algorithms": {k: {"class": type(a).__name__, "params": a.params()} for k, a in self._algs.items()},
            "problems": [{"name": p.name, "sense": p.sense, "optimum": p.optimum, "code_size": p.code_size}
                         | ({} if p.n_obj == 1 else {"indicators": self._indicator_info[p.name]})
                         for p in self.problems],
            "keys": keys,
        })

    def run(self):
        """Run every missing run, skipping finished ones with the same settings. Returns self."""
        self._meta = self.metadata()
        if self.output is not None:
            store.write_json(Path(self.output) / "metadata.json", self._meta)
        todo = []
        for t in self._tasks():
            done = (store.read_run(t["path"], t["key"]) if t["path"] is not None
                    else self._records.get((t["label"], t["problem"].name, t["run"])))
            if done is None or done["key"] != t["key"]:
                todo.append(t)
        if self.n_jobs > 1 and len(todo) > 1:
            with ProcessPoolExecutor(min(self.n_jobs, len(todo))) as pool:
                done = list(pool.map(run_task, todo))
        else:
            done = [run_task(t) for t in todo]
        if self.output is None:
            for rec in done:
                self._records[rec["label"], rec["problem"], rec["run"]] = rec
        return self

    def report(self):
        """A ``Report`` of the stored runs (from the output folder, or from memory)."""
        from .report import Report

        if self.output is not None:
            return Report.load(self.output)
        meta = getattr(self, "_meta", None) or self.metadata()
        return Report(meta, list(self._records.values()))


def _label(algorithms):
    out, count = {}, {}
    for a in algorithms:
        name = getattr(a, "short_name", type(a).__name__)
        count[name] = count.get(name, 0) + 1
        out[name if count[name] == 1 else f"{name} ({count[name]})"] = a
    return out


def indicator_setup(problem):
    """The minimized true front (or None), reference point, ideal point and scale used for
    the indicators of a multi-objective problem, or None for one objective."""
    if problem.n_obj == 1:
        return None
    m, sign = problem.n_obj, problem.comparator.sign
    front, ref = problem.info.get("pareto_front"), problem.info.get("ref_point")
    if front is None and ref is None:
        raise ValueError(f"{problem.name} has several objectives: give its true front as "
                         "info['pareto_front'] or a hypervolume reference point as info['ref_point']")
    ideal, scale = np.zeros(m), np.ones(m)
    if front is not None:
        front = sign * np.asarray(front, dtype=float).reshape(-1, m)
        ideal, nadir = front.min(axis=0), front.max(axis=0)
        scale = np.where(nadir > ideal, nadir - ideal, 1.0)  # an objective with no range is not scaled
    if ref is None:
        ref = ideal + 1.1 * scale
    else:
        ref = sign * np.asarray(ref, dtype=float).reshape(m)
    return {"front": front, "ref_point": ref, "ideal": ideal, "scale": scale, "sign": sign}


def _indicator_info(ind):
    front = ind["front"]
    return _plain({"front_points": None if front is None else len(front),
                   "front_hash": None if front is None else store.key_hash(_plain(front)),
                   "ref_point": ind["ref_point"], "ideal": ind["ideal"], "scale": ind["scale"]})


def front_indicators(ind, obj_values, violations) -> dict:
    """IGD (None without a true front) and hypervolume of a set of objective values.
    Only feasible rows with finite values count."""
    F = ind["sign"] * np.asarray(obj_values, dtype=float).reshape(-1, len(ind["ideal"]))
    F = F[(np.asarray(violations) <= 0) & np.all(np.isfinite(F), axis=1)]
    return {"igd": None if ind["front"] is None else indicators.igd(F, ind["front"]),
            "hypervolume": indicators.hypervolume((F - ind["ideal"]) / ind["scale"],
                                                  (ind["ref_point"] - ind["ideal"]) / ind["scale"])}


def run_task(task) -> dict:
    """Solve one run and save it (module level, so worker processes can call it)."""
    alg, problem = task["algorithm"], task["problem"]
    result = alg.solve(problem, seed=task["seed"])
    cp, ind = None, task.get("indicators")
    if task["checkpoints"] is not None and ind is None:
        values = result.history.at_nfe(task["checkpoints"])
        error = None if problem.optimum is None else np.abs(values - problem.optimum)
        cp = {"nfe": task["checkpoints"], "obj_value": values, "error": error}
    elif task["checkpoints"] is not None:
        h = result.history
        at = [front_indicators(ind, h.trace_obj_value[mask], h.trace_violation[mask])
              for mask in map(h.front_at, task["checkpoints"])]
        cp = {"nfe": task["checkpoints"], "igd": [a["igd"] for a in at],
              "hypervolume": [a["hypervolume"] for a in at]}
    rec = {
        "key": task["key"], "settings": task["settings"], "label": task["label"],
        "problem": problem.name, "run": task["run"], "sense": problem.sense,
        "optimum": problem.optimum, "checkpoints": cp, "result": result.to_dict(),
    }
    if ind is not None:
        rec["indicators"] = front_indicators(ind, result.front.obj_values, result.front.violations)
    rec = _plain(rec)
    if task["path"] is None:
        return rec
    store.write_json(task["path"], rec)  # return nothing, so a large experiment does not fill the memory
    return None
