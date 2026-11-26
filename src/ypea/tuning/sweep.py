"""Grid and random sweeps: many settings of one algorithm, every run of every setting."""

from __future__ import annotations

import numpy as np

from ..experiment import Experiment
from .race import _new_configs
from .spaces import Param, Space, as_space


def sweep(algorithm_class, settings, problems, runs=30, *, n=None, fixed=None, seed=0, **kwargs):
    """Run several settings of one algorithm on ``problems`` and return the ``Report``.

    settings : one of
        * a dict from parameter name to a list of values (a grid: every combination runs),
        * a list of dicts (each dict is one setting),
        * a ``Space`` (or a dict of name to ``Param``, or None for the built in space) with
          ``n`` random settings drawn from it, labeled "<name> 1", "<name> 2" and so on.
          The report's metadata holds the settings of each label.
    fixed : settings shared by every run (for example ``{"pop_size": 50}``)
    seed : seed of the runs, and of the random settings
    Other keyword arguments (``max_nfe``, ``n_jobs``, ``output`` and so on) go to
    ``Experiment``, so a sweep is parallel and resumable like any experiment.
    """
    fixed = dict(fixed or {})
    grid = isinstance(settings, dict) and not all(isinstance(v, Param) for v in settings.values())
    if grid:
        exp = Experiment.sweep(algorithm_class, {**settings, **{k: [v] for k, v in fixed.items()}}, problems,
                               runs=runs, seed=seed, **kwargs)
    elif isinstance(settings, list):
        exp = Experiment.sweep(algorithm_class, [{**fixed, **s} for s in settings], problems,
                               runs=runs, seed=seed, **kwargs)
    elif settings is None or isinstance(settings, Space | dict):
        if not (isinstance(n, int) and n >= 1):
            raise ValueError("random settings need n, a positive integer")
        sp = as_space(algorithm_class, settings)
        sp = Space({k: p for k, p in sp.params.items() if k not in fixed}, sp.fixed)
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
        drawn = _new_configs(algorithm_class, sp, fixed, {}, rng, n, set())
        algorithms = {f"{algorithm_class.short_name} {i}": algorithm_class(**fixed, **cfg)
                      for i, (cfg, _) in enumerate(drawn, start=1)}
        exp = Experiment(algorithms, problems, runs=runs, seed=seed, **kwargs)
    else:
        raise TypeError("settings must be a dict of lists, a list of dicts, a Space or None")
    return exp.run().report()
