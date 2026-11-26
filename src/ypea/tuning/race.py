"""Iterated racing for algorithm settings, as in irace.

The method follows Lopez-Ibanez, Dubois-Lacoste, Perez Caceres, Birattari and Stutzle
(2016), "The irace package: Iterated racing for automatic algorithm configuration",
Operations Research Perspectives 3, 43 to 58, with the F-Race of Birattari, Stutzle,
Paquete and Varrentrapp (2002), "A racing algorithm for configuring metaheuristics",
GECCO 2002, 11 to 18.

Why a loop of its own and not ``Experiment``: a race runs a shrinking set of settings one
instance at a time, while an experiment runs a fixed grid. Building an experiment for each
step would start a new process pool every step and write a full history file for every
run. So the race keeps one process pool for the whole tuning, stores one small file per
run (the final error and violation), and reuses the experiment's statistics (Friedman test,
post hoc test with the Holm correction) and its file helpers (atomic writes, run keys for
resuming).
"""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..experiment import stats, store
from ..result import _plain
from .spaces import as_space


@dataclass
class TuneResult:
    """The outcome of ``tune``.

    best : the best setting found (tuned parameters only)
    elites : the surviving settings of the last race, best first
    history : one dict per iteration (new and total settings, instances, runs used,
        survivors, elites and their mean ranks)
    runs : runs used in total (at most the budget)
    fixed : the settings that were not tuned
    max_nfe_per_run : the evaluation budget of each tuning run
    """

    algorithm_class: type
    best: dict
    elites: list
    history: list
    runs: int
    fixed: dict = field(default_factory=dict)
    max_nfe_per_run: int | None = None

    def algorithm(self, **overrides):
        """The tuned algorithm: ``algorithm_class(**fixed, **best, **overrides)``. Give it
        your own budget, for example ``result.algorithm(max_nfe=50_000)``."""
        return self.algorithm_class(**{**self.fixed, **self.best, **overrides})

    def to_dict(self) -> dict:
        return _plain({
            "algorithm": f"{self.algorithm_class.__module__}.{self.algorithm_class.__qualname__}",
            "best": self.best, "elites": self.elites, "history": self.history, "runs": self.runs,
            "fixed": self.fixed, "max_nfe_per_run": self.max_nfe_per_run,
        })


def tune(algorithm_class, problems, budget, space=None, max_nfe_per_run=10_000, fixed=None, seed=0,
         n_jobs=1, first_test=5, alpha=0.05, output=None) -> TuneResult:
    """Tune the settings of ``algorithm_class`` by iterated racing.

    Parameters
    ----------
    algorithm_class : an Algorithm subclass, for example ``ypea.DE``
    problems : list of Problem. Each tuning instance is a problem with a fresh seed. Use
        other problems (or other seeds of the shift and rotation) to test the result.
    budget : total number of runs (one run is one algorithm run on one instance)
    space : Space, dict of name to Param, or None for ``ypea.tuning.space(algorithm_class)``
    max_nfe_per_run : evaluation budget of each run (None keeps the algorithm's own limits)
    fixed : dict of settings that are not tuned (removed from the space)
    seed : the same seed gives the same result, for any ``n_jobs``
    n_jobs : number of worker processes (problems must then be picklable)
    first_test : number of instances before the first elimination test (at least 2)
    alpha : significance level of the Friedman test and of the Holm post hoc tests
    output : folder for one file per run. A repeated call with the same arguments reads
        finished runs instead of running them again, so an interrupted tuning resumes.

    Each run is scored by its error when the problem has an optimum, otherwise by its
    objective value (negated for maximization). Infeasible results rank after feasible ones.
    Within an instance the settings are ranked, and the ranks feed the Friedman test.

    The schedule follows irace: about 2 + log2(d) iterations for d tuned parameters, each
    with an equal share of the remaining budget; iteration j races
    share / (first_test + min(5, j)) settings. After ``first_test`` instances a Friedman
    test runs after every instance, and when it is significant, every setting that the post
    hoc test (each setting against the best one, Holm corrected, see
    ``ypea.experiment.stats.posthoc``) finds worse is dropped. irace uses Conover's post
    hoc test here. A race ends at
    2 + log2(d) survivors or when its share is used. The survivors (elites) go on to the
    next race with their results, and new settings are sampled around them: a parent is
    chosen with a probability that falls linearly with its rank, numbers come from a
    truncated normal distribution whose width shrinks each iteration, and categories from
    probabilities that move toward the parent's value. Elites cannot be dropped before the
    new settings have seen the instances that the elites have seen.
    """
    fixed = dict(fixed or {})
    problems = list(problems)
    if any(p.n_obj > 1 for p in problems):
        raise ValueError("tune races single objective problems only: it compares one final value per run. "
                         "Turn a problem into one objective with ypea.scalarize, or compare settings with "
                         "ypea.tuning.sweep, whose report gives IGD and hypervolume")
    names = [p.name for p in problems]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"problem names must be unique, got {names}")
    for name, v, low in (("budget", budget, 1), ("n_jobs", n_jobs, 1), ("first_test", first_test, 2)):
        if not (isinstance(v, int) and v >= low):
            raise ValueError(f"{name} must be an integer >= {low}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if stats.scipy_stats() is None:
        raise ImportError(stats.STATS_HINT)
    sp = as_space(algorithm_class, space)
    sp = type(sp)({k: p for k, p in sp.params.items() if k not in fixed}, sp.fixed)
    d = len(sp.params)
    if d == 0:
        raise ValueError("nothing to tune: the space is empty after removing the fixed settings")
    n_iter = n_min = int(2 + math.log2(d))
    budget_settings = {} if max_nfe_per_run is None else {"max_nfe": max_nfe_per_run}

    inst_seq, sample_seq = np.random.SeedSequence(seed).spawn(2)
    inst_rng, rng = np.random.default_rng(inst_seq), np.random.default_rng(sample_seq)
    instances = []  # (problem index, run seed), the same stream for every setting

    def instance(k):
        while len(instances) <= k:
            order = inst_rng.permutation(len(problems))
            seeds = inst_rng.integers(0, 2**32, size=len(problems))
            instances.extend((int(p), int(s)) for p, s in zip(order, seeds, strict=True))
        return instances[k]

    configs, models, results = [], [], {}  # results[(config index, instance index)] = (f, v)
    seen = set()
    pool = ProcessPoolExecutor(n_jobs) if n_jobs > 1 else None

    def run(pairs):
        tasks = []
        for c, k in pairs:
            p, s = instance(k)
            settings = {**fixed, **configs[c], **budget_settings}
            key = store.key_hash({"algorithm": f"{algorithm_class.__module__}.{algorithm_class.__qualname__}",
                                  "settings": _plain(settings), "problem": names[p], "seed": s})
            path = None if output is None else str(Path(output) / "runs" / f"{key}.json")
            tasks.append((algorithm_class, settings, problems[p], s, path, key))
        out = pool.map(run_one, tasks) if pool is not None and len(tasks) > 1 else map(run_one, tasks)
        results.update(zip(pairs, out, strict=True))

    def race(alive, protected, n_old, share):
        used, step = 0, 0
        while True:
            need = [(c, step) for c in alive if (c, step) not in results]
            if used + len(need) > share:
                break
            run(need)
            used += len(need)
            step += 1
            if step >= first_test:
                fr = stats.friedman(_rank_matrix(results, alive, step))
                if fr["p"] is not None and fr["p"] < alpha:  # None with a single setting left
                    holm = stats.posthoc(fr["ranks"], step, int(np.argmin(fr["ranks"])))["holm"]
                    alive = [c for i, c in enumerate(alive)
                             if not (holm.get(i, 1.0) < alpha and not (c in protected and step < n_old))]
                if len(alive) <= n_min:
                    break
        return alive, step, used

    try:
        elites, history, used, n_seen = [], [], 0, 0
        j = 0
        while True:
            j += 1
            share = (budget - used) / max(1, n_iter - j + 1)
            n_new = int(share // (first_test + min(5, j))) - len(elites)
            if j == 1 and n_new < 2:
                raise ValueError(f"budget {budget} is too small: the first race needs at least "
                                 f"{2 * (first_test + 1) * n_iter} runs")
            if n_new < 1:
                if budget - used > budget / 4:  # irace stops here too, but silently losing a quarter is a surprise
                    warnings.warn(f"tuning stopped with {budget - used} of {budget} runs unused: the remaining budget "
                                  f"cannot race more settings than the {len(elites)} elites. Raise the budget or "
                                  "tune fewer settings.", stacklevel=2)
                break
            new = _new_configs(algorithm_class, sp, fixed, budget_settings, rng, n_new, seen,
                               parents=[(configs[e], models[e]) for e in elites], j=j, n_iter=n_iter, d=d)
            if not new:
                break
            for cfg, model in new:
                configs.append(cfg)
                models.append(model)
            alive = elites + list(range(len(configs) - len(new), len(configs)))
            n_start = len(alive)
            alive, step, used_j = race(alive, set(elites), n_seen, min(share, budget - used))
            if step == 0:
                break
            used += used_j
            n_seen = max(n_seen, step)
            mean_ranks = stats.friedman(_rank_matrix(results, alive, step))["ranks"]
            order = sorted(range(len(alive)), key=lambda i: mean_ranks[i])  # stable: earlier settings win ties
            elites = [alive[i] for i in order[:n_min]]
            history.append({
                "iteration": j, "new": len(new), "settings": n_start,
                "instances": step, "runs": used_j, "survivors": len(alive),
                "elites": [configs[e] for e in elites], "mean_ranks": [float(mean_ranks[i]) for i in order[:n_min]],
            })
            if used_j == 0:
                break
    finally:
        if pool is not None:
            pool.shutdown()

    if not elites:
        raise ValueError("the budget allowed no complete race")
    res = TuneResult(algorithm_class, dict(configs[elites[0]]), [dict(configs[e]) for e in elites], history,
                     used, fixed, max_nfe_per_run)
    if output is not None:
        store.write_json(Path(output) / "tune.json", res.to_dict())
    return res


def _rank_matrix(results, alive, steps):
    """Ranks (1 is best) of the alive settings, one row per instance."""
    rows = []
    for k in range(steps):
        f, v = (np.array(x, dtype=float) for x in zip(*(results[c, k] for c in alive), strict=True))
        rows.append(_ranks(f, v))
    return np.array(rows)


def _ranks(f, v):
    """Feasible results by f, then infeasible ones by violation. Ties share their average rank."""
    f = np.where(np.isnan(f), np.inf, f)
    feas = v <= 0
    r = np.empty(len(f))
    r[feas] = stats.rankdata(f[feas]) if feas.any() else []
    r[~feas] = feas.sum() + (stats.rankdata(v[~feas]) if (~feas).any() else [])
    return r


def _key(cfg):
    return tuple(sorted((k, repr(v)) for k, v in cfg.items()))


def _new_configs(algorithm_class, sp, fixed, budget_settings, rng, n, seen, parents=(), j=1, n_iter=1, d=1):
    """Up to n new settings with their sampling models (sigma and category probabilities).

    Settings that the algorithm rejects are drawn again, and duplicates are skipped."""
    out, invalid, tries = [], 0, 0
    if parents:
        w = np.arange(len(parents), 0, -1, dtype=float)
        w /= w.sum()  # rank r of k elites gets (k - r + 1) / (k (k + 1) / 2)
        shrink = (1 / n) ** (1 / d)
        keep = min((j - 1) / n_iter, 0.9)  # irace's update, capped so no category becomes certain
    while len(out) < n and tries < 100 * n:
        tries += 1
        if parents:
            parent, model = parents[int(rng.choice(len(parents), p=w))]
            sigma = model["sigma"] * shrink
            probs = {name: list((1 - keep) * np.asarray(p) + keep * np.array(
                [v == parent[name] for v in sp.params[name].values], dtype=float))
                for name, p in model["probs"].items()}
            cfg = sp.neighbor(rng, parent, sigma, probs)
        else:
            sigma = 0.5
            probs = {name: [1 / len(p.values)] * len(p.values) for name, p in sp.params.items() if p.kind == "cat"}
            cfg = sp.sample(rng)
        if _key(cfg) in seen:
            continue
        try:
            algorithm_class(**{**fixed, **cfg, **budget_settings})
        except (ValueError, TypeError):
            invalid += 1
            continue
        seen.add(_key(cfg))
        out.append((cfg, {"sigma": sigma, "probs": probs}))
    if not out and invalid:
        raise ValueError(f"{invalid} sampled settings of {algorithm_class.__name__} were all invalid, "
                         "check the space and the fixed settings")
    return out


def run_one(task):
    """One run: (f, v) with f the error or the objective (smaller is better) and v the
    violation. Reads a stored run when its key matches (module level, for worker processes)."""
    algorithm_class, settings, problem, seed, path, key = task
    if path is not None:
        rec = store.read_run(path, key)
        if rec is not None:
            return rec["f"], rec["v"]
    r = algorithm_class(**settings).solve(problem, seed=seed)
    f = r.error if r.error is not None else (-1.0 if problem.sense == "max" else 1.0) * r.best.obj_value
    f = math.inf if math.isnan(f) else float(f)
    v = float(r.best.violation)
    if path is not None:
        store.write_json(path, {"key": key, "f": f, "v": v, "settings": _plain(settings),
                                "problem": problem.name, "seed": seed})
    return f, v
