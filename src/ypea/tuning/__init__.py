"""Tuning: find good settings of an algorithm for a set of problems.

* ``tune(ypea.DE, problems, budget=2000)``: iterated racing, as in irace.
* ``sweep(ypea.DE, {"crossover_prob": [0.1, 0.5, 0.9]}, problems)``: a grid, a list of
  settings or random settings, run as an experiment, returning its ``Report``.
* ``space(ypea.DE)``: the built in parameter space. Build your own with ``Space``,
  ``real``, ``integer`` and ``categorical``.

Statistical tests need SciPy (``pip install ypea[stats]``).
"""

from .race import TuneResult, tune
from .spaces import SPACES, Param, Space, categorical, integer, real, space
from .sweep import sweep

__all__ = ["SPACES", "Param", "Space", "TuneResult", "categorical", "integer", "real", "space", "sweep", "tune"]
