"""Experiments: run several algorithms on several problems, then compare them.

::

    from ypea.experiment import Experiment

    exp = Experiment({"PSO": ypea.PSO(), "DE": ypea.DE()}, problems, runs=30,
                     max_nfe=100_000, n_jobs=4, output="results/my-study")
    exp.run()                      # finished runs are skipped
    report = exp.report()          # or Report.load("results/my-study") later
    report.table(baseline="DE").to_latex("table.tex")
    report.plot_convergence(problems[0].name)
    report.save_html()

Statistical tests need SciPy (``pip install ypea[stats]``) and figures need matplotlib
(``pip install ypea[plot]``).
"""

from .report import Report
from .runner import CEC_CHECKPOINTS, Experiment
from .tables import Table

__all__ = ["CEC_CHECKPOINTS", "Experiment", "Report", "Table"]
