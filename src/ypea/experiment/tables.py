"""Comparison tables and their export to CSV, LaTeX, Markdown, HTML and pandas.

Marks of the rank sum test: "+" (better than the control), "=" (no significant
difference) and a minus sign (worse). The minus sign is U+2212 in CSV, Markdown and
HTML, and ``$-$`` in LaTeX.
"""

from __future__ import annotations

import csv
import html
import io
from pathlib import Path

import numpy as np

from . import stats as st

STATS = {"mean": np.mean, "std": lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0,
         "best": None, "worst": None, "median": np.median}
LABELS = {"mean": "Mean", "std": "Std", "best": "Best", "worst": "Worst", "median": "Median"}
MINUS = "−"


def describe(v, smaller_is_better=True) -> dict:
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return dict.fromkeys(STATS, np.nan)
    lo, hi = float(np.min(v)), float(np.max(v))
    return {"mean": float(np.mean(v)), "std": float(STATS["std"](v)),
            "best": lo if smaller_is_better else hi, "worst": hi if smaller_is_better else lo,
            "median": float(np.median(v))}


class Table:
    """A table of results. ``header`` is a list of column names, ``rows`` a list of rows of
    cells (dicts with ``text``, ``bold`` and ``mark``), ``groups`` the row indices where a
    new group starts (drawn as rules), ``notes`` lines printed under the table, and
    ``data`` the numbers behind it."""

    def __init__(self, header, rows, groups=(), notes=(), data=None):
        self.header, self.rows, self.groups = list(header), rows, set(groups)
        self.notes, self.data = list(notes), data or {}

    def _cells(self, bold, marks, esc=lambda s: s):
        out = []
        for row in self.rows:
            line = []
            for c in row:
                t = esc(c["text"])
                if c.get("bold") and bold:
                    t = bold.format(t)
                if c.get("mark"):
                    t = f"{t} {marks[c['mark']]}"
                line.append(t)
            out.append(line)
        return out

    def to_csv(self, path=None) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.header)
        w.writerows(self._cells(None, {"+": "+", "=": "=", "-": MINUS}))
        return _save(buf.getvalue(), path)

    def to_markdown(self, path=None) -> str:
        def esc(s):
            return s.replace("|", "\\|")
        lines = ["| " + " | ".join(map(esc, self.header)) + " |",
                 "|" + "|".join([":---"] * 2 + ["---:"] * (len(self.header) - 2)) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in self._cells("**{}**", {"+": "+", "=": "=", "-": MINUS}, esc)]
        lines += ["", *self.notes] if self.notes else []
        return _save("\n".join(lines) + "\n", path)

    def to_latex(self, path=None, caption=None, label=None) -> str:
        """A booktabs table (needs ``\\usepackage{booktabs}``)."""
        cols = "ll" + "r" * (len(self.header) - 2)
        lines = ["\\begin{table}[htbp]", "\\centering"]
        if caption:
            lines.append(f"\\caption{{{_tex(caption)}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines += [f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
                  " & ".join(f"\\textbf{{{_tex(h)}}}" for h in self.header) + " \\\\", "\\midrule"]
        for i, r in enumerate(self._cells("\\textbf{{{}}}", {"+": "$+$", "=": "$=$", "-": "$-$"}, _tex)):
            if i in self.groups and i:
                lines.append("\\midrule")
            lines.append(" & ".join(r) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines += [f"\\par\\smallskip\\footnotesize {_tex(n)}" for n in self.notes]
        lines.append("\\end{table}")
        return _save("\n".join(lines) + "\n", path)

    def to_html(self, path=None) -> str:
        e = html.escape
        lines = ["<table>", "<thead><tr>" + "".join(f"<th>{e(h)}</th>" for h in self.header) + "</tr></thead>",
                 "<tbody>"]
        for i, r in enumerate(self._cells("<b>{}</b>", {"+": "+", "=": "=", "-": MINUS}, e)):
            cls = ' class="group"' if i in self.groups and i else ""
            lines.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        lines.append("</tbody></table>")
        lines += [f"<p class=\"note\">{e(n)}</p>" for n in self.notes]
        return _save("\n".join(lines) + "\n", path)

    def to_dataframe(self):
        """A pandas DataFrame of the cell texts (needs pandas)."""
        import pandas as pd

        return pd.DataFrame(self._cells(None, {"+": "+", "=": "=", "-": MINUS}), columns=self.header)

    def __str__(self):
        return self.to_markdown()


def _tex(s):
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}"),
                 (MINUS, "$-$")):
        s = s.replace(a, b)
    return s


def _save(text, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    return text


def comparison_table(samples, labels, problems, smaller, stat="mean", stats=tuple(STATS), test="wilcoxon",
                     baseline=None, alpha=0.05, fmt=".3e", metric="error"):
    """Build the comparison table.

    samples[problem][label] is the array of run values. smaller[problem] is True when
    smaller values are better. The control of the tests is ``baseline``, or the best
    Friedman ranked algorithm when ``baseline`` is None.
    """
    if stat not in STATS or not set(stats) <= set(STATS):
        raise ValueError(f"stat must be one of {list(STATS)}")
    if test not in ("wilcoxon", None):
        raise ValueError("test must be 'wilcoxon' or None")
    if baseline is not None and baseline not in labels:
        raise ValueError(f"baseline {baseline!r} is not one of {labels}")
    desc = {p: {a: describe(samples[p][a], smaller[p]) for a in labels} for p in problems}
    # Friedman ranks over problems, from the chosen statistic; maximization becomes minimization
    keys = [[desc[p][a][stat] * (1 if smaller[p] else -1) for a in labels] for p in problems]
    fr = st.friedman(keys)
    best_ranked = labels[int(np.argmin(fr["ranks"]))]
    control = baseline or best_ranked
    ci = labels.index(control)
    have_scipy = st.scipy_stats() is not None
    marks = {}
    if test and have_scipy:
        for p in problems:
            for a in labels:
                if a != control and len(samples[p][a]) and len(samples[p][control]):
                    marks[p, a] = st.rank_sum(samples[p][a], samples[p][control], alpha, smaller[p])[0]

    def num(x):
        return "n/a" if x is None or not np.isfinite(x) else format(x, fmt)

    rows, groups = [], []
    for p in problems:
        groups.append(len(rows))
        for s in stats:
            vals = np.array([desc[p][a][s] for a in labels])
            key = vals if smaller[p] else -vals
            win = np.nanmin(key) if s != "std" and np.isfinite(key).any() else None
            row = [{"text": p if s == stats[0] else ""}, {"text": LABELS[s]}]
            for a, v, kv in zip(labels, vals, key, strict=True):
                row.append({"text": num(v), "bold": win is not None and kv == win,
                            "mark": marks.get((p, a)) if s == stat else None})
            rows.append(row)
    notes = []
    groups.append(len(rows))
    if marks:
        wtl = {a: [sum(marks.get((p, a)) == m for p in problems) for m in "+=-"] for a in labels if a != control}
        rows.append([{"text": f"+/=/{MINUS}"}, {"text": f"vs {control}"}]
                    + [{"text": "/".join(map(str, wtl[a])) if a in wtl else "control"} for a in labels])
        notes.append(f"Wilcoxon rank sum test against {control} at alpha = {alpha}: + better, = no "
                     f"significant difference, {MINUS} worse.")
    elif test and not have_scipy:
        notes.append(f"No rank sum test: {st.STATS_HINT}.")
    ph = st.posthoc(fr["ranks"], fr["n"], ci) if len(labels) > 1 else None
    rows.append([{"text": "Friedman"}, {"text": "Avg. rank"}]
                + [{"text": f"{r:.2f}", "bold": r == fr["ranks"].min()} for r in fr["ranks"]])
    if ph and fr["n"] > 1:
        rows.append([{"text": ""}, {"text": "Holm APV"}]
                    + [{"text": "control" if i == ci else f"{ph['holm'][i]:.3g}"} for i in range(len(labels))])
    if fr["p"] is not None:
        idd = (f" Iman and Davenport: F = {fr['f']:.4g}, p = {fr['p_f']:.3g}." if np.isfinite(fr["f"]) else "")
        notes.append(f"Friedman test on the {stat} {metric} over {fr['n']} problems: chi2 = {fr['chi2']:.4g}, "
                     f"p = {fr['p']:.3g}.{idd} Post hoc z test against {control} with Holm adjusted p values "
                     f"(Derrac et al. 2011).")
    data = {"describe": desc, "marks": marks, "friedman": fr, "posthoc": ph, "control": control,
            "labels": list(labels), "problems": list(problems)}
    return Table(["Problem", "Statistic", *labels], rows, groups, notes, data)
