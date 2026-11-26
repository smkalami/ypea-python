"""Reports of an experiment: tables, figures and a one page HTML summary."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np

from . import store
from .tables import comparison_table

METRICS = ("error", "obj_value", "igd", "hypervolume")
NAMES = {"igd": "IGD", "hypervolume": "hypervolume"}


class Report:
    """Results of an experiment, built from its run records.

    Make one with ``Experiment.report()``, or later from the output folder alone with
    ``Report.load(path)``.

    ``metric="error"`` uses ``|f - optimum|`` (smaller is better). On problems without a
    known optimum it falls back to the objective value, compared by the problem sense.

    With several objectives the metrics are ``"igd"`` (smaller is better) and
    ``"hypervolume"`` (larger is better), as stored by the runner (see ``Experiment``).
    There, ``metric="error"`` means IGD, or the hypervolume on problems without a true front.

    ``zero_below``: errors below this value count as 0. The CEC rules use 1e-8, so set
    ``report.zero_below = 1e-8`` when comparing with published CEC results. The default
    None keeps the raw values.
    """

    zero_below = None

    def __init__(self, metadata, records, output=None):
        self.metadata, self.output = metadata, output
        self.labels = list(metadata["algorithms"])
        self.problems = [p["name"] for p in metadata["problems"]]
        self._problem = {p["name"]: p for p in metadata["problems"]}
        self._runs = {}
        for r in sorted(records, key=lambda r: r["run"]):
            self._runs.setdefault((r["label"], r["problem"]), []).append(r)

    @classmethod
    def load(cls, path):
        meta, records = store.load(path)
        return cls(meta, records, output=Path(path))

    def records(self, label, problem) -> list:
        """The run records of one algorithm on one problem, in run order."""
        return self._runs.get((label, problem), [])

    def _key(self, problem, metric):
        """The stored value behind ``metric`` on ``problem``: "error", "obj_value", "igd" or
        "hypervolume"."""
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {list(METRICS)}")
        p = self._problem[problem]
        if isinstance(p["sense"], str):
            if metric in ("igd", "hypervolume"):
                raise ValueError(f"{problem} has one objective: use metric 'error' or 'obj_value'")
            return "error" if metric == "error" and p["optimum"] is not None else "obj_value"
        has_front = (p.get("indicators") or {}).get("front_points") is not None
        if metric == "obj_value" or (metric == "igd" and not has_front):
            raise ValueError(f"{problem} has several objectives and "
                             + ("no true front: use metric 'hypervolume'" if not has_front else
                                "no single objective value: use metric 'igd' or 'hypervolume'"))
        return metric if metric != "error" else "igd" if has_front else "hypervolume"

    def smaller_is_better(self, problem, metric="error") -> bool:
        key = self._key(problem, metric)
        return key in ("error", "igd") or (key == "obj_value" and self._problem[problem]["sense"] == "min")

    def values(self, label, problem, metric="error") -> np.ndarray:
        """Final values of the runs of one algorithm on one problem."""
        key = self._key(problem, metric)
        if key in ("igd", "hypervolume"):
            return np.array([float(r["indicators"][key]) for r in self.records(label, problem)])
        v = np.array([float(r["result"][key] if key == "error" else r["result"]["best"]["obj_value"])
                      for r in self.records(label, problem)])
        return self._floor(v) if key == "error" else v

    def _floor(self, errors):
        if self.zero_below is None:
            return errors
        return np.where(errors < self.zero_below, 0.0, errors)

    def curves(self, label, problem, metric="error", grid=None):
        """Best value so far of each run at the evaluations in ``grid`` (default: 200 points
        up to the largest NFE). Returns (grid, array of shape (runs, len(grid))).

        With several objectives the values are IGD or hypervolume at the checkpoints (the
        default grid), and each grid point takes the last checkpoint at or before it."""
        recs = self.records(label, problem)
        if grid is None:
            grid = self._grid(problem)
        key = self._key(problem, metric)
        opt = self._problem[problem]["optimum"] if key == "error" else None
        rows = []
        for r in recs:
            if key in ("igd", "hypervolume"):
                cp = r["checkpoints"]
                if cp is None:
                    raise ValueError(f"the runs on {problem} have no checkpoints (they need max_nfe)")
                t_nfe, t_f = np.array(cp["nfe"], dtype=float), np.array([float(v) for v in cp[key]])
            else:
                t_nfe = np.array(r["result"]["trace"]["nfe"], dtype=float)
                t_f = np.array([float(v) for v in r["result"]["trace"]["obj_value"]])
            idx = np.searchsorted(t_nfe, grid, side="right") - 1
            y = np.where(idx >= 0, t_f[np.maximum(idx, 0)], np.nan)
            rows.append(self._floor(np.abs(y - opt)) if opt is not None else y)
        return grid, np.array(rows).reshape(len(rows), len(grid))

    def _grid(self, problem, points=200):
        if not isinstance(self._problem[problem]["sense"], str):
            # several objectives: the checkpoints of the longest run
            cps = [r["checkpoints"]["nfe"] for a in self.labels for r in self.records(a, problem)
                   if r["checkpoints"] is not None]
            return np.array(max(cps, key=max) if cps else [1])
        top = max([r["result"]["nfe"] for a in self.labels for r in self.records(a, problem)] or [1])
        return np.unique(np.round(np.linspace(1, top, points)).astype(int))

    def table(self, stat="mean", metric="error", test="wilcoxon", baseline=None, alpha=0.05, fmt=".3e",
              stats=("mean", "std", "best", "worst", "median")):
        """Comparison table: statistics per algorithm and problem with the best in bold,
        Wilcoxon rank sum marks and a win/tie/loss row against the control (``baseline``,
        or the best ranked algorithm), and Friedman average ranks of ``stat`` with Holm
        adjusted post hoc p values (Derrac et al. 2011). ``test=None`` skips the rank sum
        test. Returns a ``Table`` (``to_csv``, ``to_latex``, ``to_markdown``, ``to_html``,
        ``to_dataframe``).
        """
        samples = {p: {a: self.values(a, p, metric) for a in self.labels} for p in self.problems}
        smaller = {p: self.smaller_is_better(p, metric) for p in self.problems}
        return comparison_table(samples, self.labels, self.problems, smaller, stat=stat, stats=tuple(stats),
                                test=test, baseline=baseline, alpha=alpha, fmt=fmt, metric=self._shown(metric))

    def _shown(self, metric):
        """The metric's name in notes and titles: IGD or hypervolume when every problem has
        several objectives and ``metric="error"`` resolves to the same one."""
        keys = {self._key(p, metric) for p in self.problems}
        if metric == "error" and len(keys) == 1 and keys <= {"igd", "hypervolume"}:
            metric = keys.pop()
        return NAMES.get(metric, metric)

    def _ylabel(self, problem, metric, final=False):
        key = self._key(problem, metric)
        name = {"error": "error", "obj_value": "objective value"}.get(key, NAMES.get(key))
        if final:
            return f"Final {name}"
        return {"error": "Error", "obj_value": "Best objective value", "igd": "IGD"}.get(key, "Hypervolume")

    def plot_convergence(self, problem, metric="error", center="median", log=None, ax=None):
        """Median (or mean) best value against evaluations, with a 25 to 75 percent band."""
        from .. import plot

        grid = self._grid(problem)
        # a run that stopped early keeps its last value up to the end of the grid
        curves = {a: self.curves(a, problem, metric, grid)[1] for a in self.labels if self.records(a, problem)}
        return plot.band(grid, curves, center=center, log=log, ax=ax, ylabel=self._ylabel(problem, metric),
                         title=problem)

    def plot_box(self, problem, metric="error", log=None, ax=None):
        """Box plots of the final values of each algorithm on one problem."""
        from .. import plot

        samples = {a: self.values(a, problem, metric) for a in self.labels if self.records(a, problem)}
        return plot.boxes(samples, log=log, ax=ax, ylabel=self._ylabel(problem, metric, True), title=problem)

    def plot_ranks(self, stat="mean", metric="error", ax=None):
        """Bar chart of the Friedman average ranks."""
        from .. import plot

        ranks = self.table(stat, metric, test=None).data["friedman"]["ranks"]
        return plot.rank_bars(dict(zip(self.labels, ranks, strict=True)), ax=ax)

    def save_html(self, path=None, stat="mean", metric="error", baseline=None, figures=True) -> Path:
        """One self contained HTML page with the settings, the table and the figures
        (PNG images embedded in the page). Figures need matplotlib and are skipped without
        it. ``path`` defaults to ``report.html`` in the output folder."""
        if path is None:
            if self.output is None:
                raise ValueError("give a path (this report has no output folder)")
            path = self.output / "report.html"
        e = html.escape
        m = self.metadata
        settings = {k: m.get(k) for k in ("runs", "max_nfe", "max_iter", "seed", "seeds", "started",
                                          "platform")}
        settings["versions"] = ", ".join(f"{k} {v}" for k, v in m.get("versions", {}).items())
        parts = ["<h1>Experiment report</h1>", "<h2>Settings</h2>", "<table>"]
        parts += [f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in settings.items()]
        parts.append("</table><h2>Algorithms</h2><table>")
        for a, info in m["algorithms"].items():
            parts.append(f"<tr><th>{e(a)}</th><td><code>{e(info['class'])}("
                         f"{e(json.dumps(info['params'], default=str)[1:-1])})</code></td></tr>")
        parts.append("</table>")
        parts += [f"<h2>Results ({e(stat)} {e(self._shown(metric))})</h2>",
                  self.table(stat, metric, baseline=baseline).to_html()]
        if figures and _has_matplotlib():
            import matplotlib.pyplot as plt

            # draw, encode and close one figure at a time
            imgs = [lambda: self.plot_ranks(stat, metric)]
            for p in self.problems:
                imgs += [lambda p=p: self.plot_convergence(p, metric), lambda p=p: self.plot_box(p, metric)]
            parts.append("<h2>Figures</h2><div class=\"figs\">")
            for make in imgs:
                ax = make()
                buf = io.BytesIO()
                ax.figure.savefig(buf, format="png", dpi=150)
                plt.close(ax.figure)
                parts.append(f'<img alt="figure" src="data:image/png;base64,'
                             f'{base64.b64encode(buf.getvalue()).decode()}">')
            parts.append("</div>")
        page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Experiment report"
                f"</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>\n")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page, encoding="utf-8")
        return path


def _has_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


CSS = """body{font-family:system-ui,sans-serif;margin:2em auto;max-width:72em;padding:0 1em;color:#222;
background:#fff}table{border-collapse:collapse;margin:1em 0;font-size:.9em}th,td{padding:.25em .6em;
text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child,td:nth-child(2){text-align:left}
tr.group td{border-top:2px solid #888}.note{font-size:.85em;color:#555}.figs img{width:32em;max-width:100%}"""
