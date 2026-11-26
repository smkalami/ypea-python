"""Compare a few algorithms on CEC 2017 at dimension 10, and write a report.

A small version of a benchmark study: 3 algorithms, 4 functions (unimodal, multimodal,
hybrid and composition), 3 runs each and 3 000 evaluations per run. The CEC 2017 rules ask for
51 runs and 10 000 D evaluations. Run it as

    python experiment_cec2017.py [output folder]

The output folder gets one JSON file per run, ``metadata.json``, the tables (CSV, LaTeX
and Markdown) and ``report.html`` with the figures (needs matplotlib). Run it again and
the finished runs are skipped. The rank sum marks need SciPy (``pip install ypea[stats]``).
"""

import sys
from pathlib import Path

import ypea
from ypea.experiment import Experiment

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/experiment_cec2017")

problems = ypea.benchmarks.cec2017(10, functions=[1, 5, 11, 21])
algorithms = {"DE": ypea.DE(), "PSO": ypea.PSO(), "L-SHADE": ypea.LSHADE()}

exp = Experiment(algorithms, problems, runs=3, max_nfe=3_000, seed=2026, output=out)
exp.run()  # runs already in the output folder are skipped

report = exp.report()  # or later: ypea.Report.load(out)
report.zero_below = 1e-8  # the CEC rule: errors below 1e-8 count as 0

table = report.table(stat="mean", baseline="L-SHADE")
print(table.to_markdown())
table.to_csv(out / "table.csv")
table.to_latex(out / "table.tex", caption="Mean error on CEC 2017, D = 10", label="tab:cec2017")
table.to_markdown(out / "table.md")

try:
    import matplotlib  # noqa: F401
except ImportError:
    sys.exit(0)

import matplotlib.pyplot as plt  # noqa: E402

ax = report.plot_convergence(problems[1].name)
ax.figure.savefig(out / "convergence_f5.pdf")
plt.close(ax.figure)
print("saved", report.save_html())
