"""Make the figures of the documentation from real runs (fixed seeds).

Run from the project root, with matplotlib installed:

    python docs/figures/make_figures.py            # every figure
    python docs/figures/make_figures.py tsp front  # only these

The figures go to docs/assets/figures/. They are committed, so building the docs does not
need matplotlib. Small budgets and one process keep it quick and light on memory.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ypea  # noqa: E402
from ypea import benchmarks as B  # noqa: E402
from ypea import problems as P  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets" / "figures"
plt.rcParams.update({
    "svg.hashsalt": "ypea", "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "axes.titlesize": 10, "legend.fontsize": 9, "figure.dpi": 100,
})
FIGURES = {}


def figure(func):
    FIGURES[func.__name__] = func
    return func


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    # no dates or creator tags, so the files only change when the figure does
    svg = {"Date": None, "Creator": None, "Format": None, "Type": None}
    meta = svg if name.endswith(".svg") else {"Software": None}
    fig.savefig(path, bbox_inches="tight", dpi=110, metadata=meta)
    if os.environ.get("FIGURE_PREVIEW"):  # PNG copies to look at, in a folder of your choice
        fig.savefig(Path(os.environ["FIGURE_PREVIEW"]) / (path.stem + ".png"), bbox_inches="tight", dpi=80)
    plt.close(fig)
    print(path.relative_to(OUT.parent.parent.parent), f"{path.stat().st_size / 1024:.0f} KB")


# ---------------------------------------------------------------- problems and encoding

@figure
def encoding():
    rng = np.random.default_rng(3)
    rows = [
        (ypea.Real("x", 4, -5, 5), "Real('x', 4, lb=-5, ub=5)"),
        (ypea.Integer("k", 4, 0, 9), "Integer('k', 4, lb=0, ub=9)"),
        (ypea.Binary("b", 4), "Binary('b', 4)"),
        (ypea.Permutation("p", 5), "Permutation('p', 5)"),
        (ypea.Selection("s", 6, 3), "Selection('s', 6, 3)"),
        (ypea.Partition("q", 6, 2), "Partition('q', 6, 2)"),
        (ypea.FixedSum("w", 4), "FixedSum('w', 4)"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-len(rows) + 0.3, 1.1)
    ax.axis("off")
    cmap = plt.get_cmap("Blues")
    ax.text(0, 0.75, "Variable", weight="bold")
    ax.text(3.0, 0.75, "Codes in [0, 1]", weight="bold")
    ax.text(7.3, 0.75, "Decoded value", weight="bold")
    for i, (var, label) in enumerate(rows):
        y = -i
        code = rng.random((1, var.code_size)).round(2)
        value = var.decode(code)
        ax.text(0, y, label, va="center", family="monospace", fontsize=9)
        for j, c in enumerate(code[0]):
            x0 = 3.0 + j * 0.52
            ax.add_patch(plt.Rectangle((x0, y - 0.3), 0.48, 0.6, color=cmap(0.15 + 0.7 * c)))
            ax.text(x0 + 0.24, y, f"{c:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if c > 0.55 else "black")
        ax.annotate("", (7.15, y), (3.0 + var.code_size * 0.52 + 0.05, y),
                    arrowprops={"arrowstyle": "->", "color": "0.4"})
        ax.text(7.3, y, _show(value), va="center", family="monospace", fontsize=9)
    save(fig, "encoding.svg")


def _show(value):
    if isinstance(value, list):  # partition: one list of groups per row
        return "  ".join("{" + ", ".join(map(str, g)) + "}" for g in value[0])
    v = np.asarray(value[0])
    if v.dtype.kind == "f":
        return "[" + ", ".join(f"{a:.2f}" for a in v) + "]"
    return "[" + ", ".join(map(str, v)) + "]"


@figure
def structure_example():
    """Convergence of the getting started example (GA on a circle TSP)."""
    problem = P.circle_tsp(20, seed=1)
    result = ypea.GA(max_nfe=20_000, space="native").solve(problem, seed=1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    ypea.plot.convergence(result, ax=a)
    a.set_title("Best tour length")
    P.plot_tour(problem, result, ax=b)
    b.set_title(f"Best tour, length {result.best.obj_value:.2f}")
    b.set_aspect("equal")
    save(fig, "getting-started-tsp.svg")


# ---------------------------------------------------------------- results and plots

def _rastrigin10():
    return ypea.Problem(ypea.Real("x", 10, -5.12, 5.12), lambda s: B.rastrigin(s.x), vectorized=True,
                        name="rastrigin", optimum=0.0)


@figure
def convergence():
    problem = _rastrigin10()
    results = [alg.solve(problem, seed=1) for alg in (ypea.DE(max_nfe=20_000), ypea.PSO(max_nfe=20_000))]
    ax = ypea.plot.convergence(*results, error=True)
    ax.set_title("Rastrigin, 10 dimensions")
    ax.figure.set_size_inches(6, 3.6)
    save(ax.figure, "convergence.svg")


@figure
def convergence_many():
    problem = _rastrigin10()
    algs = [ypea.GA(max_nfe=20_000), ypea.PSO(max_nfe=20_000), ypea.DE(max_nfe=20_000),
            ypea.CMAES(max_nfe=20_000), ypea.LSHADE(max_nfe=20_000), ypea.TLBO(max_nfe=20_000)]
    results = [alg.solve(problem, seed=1) for alg in algs]
    ax = ypea.plot.convergence(*results, error=True)
    ax.set_title("Six algorithms on Rastrigin, 10 dimensions (seed 1)")
    ax.figure.set_size_inches(7, 4)
    save(ax.figure, "convergence-many.svg")


def _small_report():
    problems = [B.problem("rastrigin", 10, shift=True), B.problem("rosenbrock", 10, shift=True)]
    algs = {"DE": ypea.DE(), "PSO": ypea.PSO(), "CMA-ES": ypea.CMAES()}
    exp = ypea.Experiment(algs, problems, runs=10, max_nfe=10_000, seed=1)
    return exp.run().report()


@figure
def experiment():
    rep = _small_report()
    p = rep.problems[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    rep.plot_convergence(p, ax=axes[0])
    axes[0].set_title("Rastrigin (shifted, d = 10): median, quartiles")
    rep.plot_box(p, ax=axes[1])
    axes[1].set_title("Rastrigin (shifted, d = 10): final errors")
    rep.plot_ranks(ax=axes[2])
    axes[2].set_title("Average Friedman rank (2 problems)")
    fig.tight_layout()
    save(fig, "experiment.svg")


# ---------------------------------------------------------------- benchmarks and problems

@figure
def benchmarks():
    names = ["sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel_2_26",
             "levy", "michalewicz", "styblinski_tang", "alpine1", "salomon", "expanded_schaffer_f6"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.4))
    for ax, name in zip(axes.flat, names, strict=True):
        prob = B.problem(name, 2)
        var = prob.vars[0]
        lb, ub = float(np.min(var.lb)), float(np.max(var.ub))
        if name in ("rastrigin", "ackley", "griewank", "levy", "alpine1", "salomon", "expanded_schaffer_f6"):
            lb, ub = max(lb, -10.0), min(ub, 10.0)  # zoom in, so the many local minima show
        g = np.linspace(lb, ub, 300)
        xx, yy = np.meshgrid(g, g)
        z = getattr(B, name)(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        if name in ("sphere", "rosenbrock", "griewank"):
            z = np.log10(z - z.min() + 1e-3)
        ax.contourf(xx, yy, z, levels=30, cmap="viridis")
        ax.set_title(name)
        ax.set_xticks([lb, 0, ub] if lb < 0 < ub else [lb, ub])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "benchmarks.png")


@figure
def tsp():
    problem = P.random_tsp(50, seed=2)
    result = ypea.GA(max_nfe=50_000, space="native").solve(problem, seed=1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))
    P.plot_tour(problem, np.random.default_rng(0).permutation(50), ax=a)
    a.set_title("A random tour")
    P.plot_tour(problem, result, ax=b)
    b.set_title(f"GA in the native space, length {result.best.obj_value:.1f}")
    for ax in (a, b):
        ax.set_aspect("equal")
    save(fig, "tsp.svg")


# ---------------------------------------------------------------- constraints

@figure
def constraints():
    def objective(s):
        x, y = s.x[..., 0], s.x[..., 1]
        f = (x - 2) ** 2 + (y - 1) ** 2
        g = np.stack([x ** 2 - y, x + y - 2], axis=-1)  # x^2 <= y and x + y <= 2
        return f, g
    problem = ypea.Problem(ypea.Real("x", 2, -2, 3), objective, vectorized=True)
    seen = []
    result = ypea.DE(max_iter=40, pop_size=30).solve(
        problem, seed=1, callbacks=[lambda a: seen.append(a.problem.vars[0].decode(a.pop.X).copy())
                                    if a.iter in (1, 8) else None])
    g = np.linspace(-2, 3, 300)
    xx, yy = np.meshgrid(g, g)
    f = (xx - 2) ** 2 + (yy - 1) ** 2
    feasible = (xx ** 2 <= yy) & (xx + yy <= 2)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.contour(xx, yy, f, levels=15, colors="0.75", linewidths=0.7)
    ax.contourf(xx, yy, feasible, levels=[0.5, 1.5], colors=["#9ecae1"], alpha=0.6)
    ax.scatter(*seen[0].T, s=14, color="0.45", label="first population")
    ax.scatter(*seen[-1].T, s=14, color="#d62728", label="population after 8 iterations")
    best = result.best.solution.x
    ax.scatter(*best, marker="*", s=180, color="black", label=f"best ({best[0]:.3f}, {best[1]:.3f})", zorder=3)
    ax.scatter(2, 1, marker="x", s=60, color="0.3", label="unconstrained minimum (2, 1)")
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect("equal")
    ax.set_title("DE (40 iterations), feasible region in blue: x² ≤ y, x + y ≤ 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save(fig, "constraints.svg")


# ---------------------------------------------------------------- several objectives

@figure
def fronts():
    names = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "dtlz7"]
    fig = plt.figure(figsize=(12, 6.6))
    for i, name in enumerate(names):
        m = 3 if name.startswith("dtlz") else 2
        ax = fig.add_subplot(2, 4, i + 1 if i < 3 else i + 2, projection="3d" if m == 3 else None)
        front = B.pareto_front(name, 600, m=m) if m == 3 else B.pareto_front(name, 600)
        ypea.plot.front(front, ax=ax)
        ax.set_title(name.upper())
    for i, name in enumerate(["dtlz1", "dtlz2"]):
        ax = fig.add_subplot(2, 4, 4 if i == 0 else 8, projection="3d")
        ypea.plot.front(B.pareto_front(name, 600, m=3), ax=ax)
        ax.set_title(name.upper())
    fig.tight_layout()
    save(fig, "pareto-fronts.png")


@figure
def front():
    problem = B.multi_problem("zdt1", front_points=500)
    result = ypea.NSGAII(max_nfe=20_000).solve(problem, seed=1)
    ax = ypea.plot.front(result.front.obj_values, true_front=problem.info["pareto_front"],
                         labels=["NSGA-II, 20 000 evaluations"])
    ax.set_title("ZDT1")
    ax.figure.set_size_inches(5, 4)
    save(ax.figure, "front.svg")


@figure
def indicators():
    rng = np.random.default_rng(4)
    f1 = np.sort(rng.random(8))
    pts = np.column_stack([f1, 1 - np.sqrt(f1) + 0.08 + 0.05 * rng.random(8)])
    true = B.pareto_front("zdt1", 200)
    ref = np.array([1.1, 1.1])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    # hypervolume: the union of the boxes between each point and the reference point
    order = np.argsort(pts[:, 0])
    p = pts[order]
    xs = np.r_[p[:, 0], ref[0]]
    for i in range(len(p)):
        a.add_patch(plt.Rectangle((xs[i], p[i, 1]), xs[i + 1] - xs[i], ref[1] - p[i, 1], color="#9ecae1"))
    a.scatter(*p.T, color="#d62728", zorder=3, label="front found")
    a.scatter(*ref, marker="s", color="black", label="reference point")
    a.plot(*true.T, color="0.5", lw=1, label="true front")
    hv = ypea.indicators.hypervolume(pts, ref)
    a.set_title(f"Hypervolume (blue area) = {hv:.3f}")
    a.legend(fontsize=8)
    # IGD: mean distance from each true front point to the nearest point found
    sample = true[::20]
    d = np.linalg.norm(sample[:, None] - pts[None], axis=-1)
    near = pts[d.argmin(axis=1)]
    for s, n in zip(sample, near, strict=True):
        b.plot([s[0], n[0]], [s[1], n[1]], color="0.6", lw=0.8)
    b.plot(*true.T, color="0.5", lw=1, label="true front")
    b.scatter(*sample.T, s=12, color="0.3", label="true front sample")
    b.scatter(*pts.T, color="#d62728", zorder=3, label="front found")
    b.set_title(f"IGD (mean gray line length) = {ypea.indicators.igd(pts, true):.3f}")
    b.legend(fontsize=8)
    for ax in (a, b):
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
    save(fig, "indicators.svg")


# ---------------------------------------------------------------- operators

@figure
def operators():
    from ypea.operators import SBX, PolynomialMutation
    rng = np.random.default_rng(1)
    n = 20_000
    x1, x2 = np.full((n, 1), 0.3), np.full((n, 1), 0.7)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for eta in (2, 15):
        child = np.concatenate(SBX(eta=eta, prob=1.0, prob_var=1.0)(x1, x2, rng))
        a.hist(child[:, 0], bins=120, range=(0, 1), histtype="step", density=True, label=f"eta = {eta}")
    a.axvline(0.3, color="0.5", ls="--", lw=0.8)
    a.axvline(0.7, color="0.5", ls="--", lw=0.8)
    a.set_title("SBX children of the parents 0.3 and 0.7")
    a.legend()
    x = np.full((n, 1), 0.3)
    for eta in (5, 20):
        child = PolynomialMutation(prob=1.0, eta=eta)(x, rng)
        b.hist(child[:, 0], bins=120, range=(0, 1), histtype="step", density=True, label=f"eta = {eta}")
    b.axvline(0.3, color="0.5", ls="--", lw=0.8)
    b.set_title("Polynomial mutation of 0.3")
    b.legend()
    save(fig, "operators.svg")


# ---------------------------------------------------------------- one figure per algorithm

SNAPSHOTS = (1, 5, 20, 50)


def _search_snapshots(make, name):
    var = ypea.Real("x", 2, -5.12, 5.12)
    problem = ypea.Problem(var, lambda s: B.rastrigin(s.x), vectorized=True, name="rastrigin", optimum=0.0)
    seen = {}

    def keep(alg):
        if alg.iter in SNAPSHOTS:
            seen[alg.iter] = var.decode(alg.pop.X).copy()
    result = make(max_iter=SNAPSHOTS[-1]).solve(problem, seed=1, callbacks=[keep])
    g = np.linspace(-5.12, 5.12, 250)
    xx, yy = np.meshgrid(g, g)
    z = B.rastrigin(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    fig = plt.figure(figsize=(8, 4.4))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6], wspace=0.35)
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)] + [fig.add_subplot(grid[:, 2])]
    for ax, it in zip(axes, SNAPSHOTS, strict=False):
        ax.contourf(xx, yy, z, levels=12, cmap="Blues_r", alpha=0.35)
        if it in seen:
            ax.scatter(*seen[it].T, s=16, color="#d62728", edgecolors="white", linewidths=0.4)
        ax.scatter(0, 0, marker="x", color="black", s=30)
        ax.set_title(f"iteration {it}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    ypea.plot.convergence(result, ax=axes[-1], error=True)
    axes[-1].set_title("best error")
    if axes[-1].get_legend():
        axes[-1].get_legend().remove()
    fig.tight_layout()
    save(fig, f"algorithm-{name}.png")


def _front_snapshot(make, name):
    problem = B.multi_problem("zdt1", dim=10, front_points=500)
    result = make(max_nfe=10_000).solve(problem, seed=1)
    ax = ypea.plot.front(result.front.obj_values, true_front=problem.info["pareto_front"],
                         labels=[f"{result.algorithm}, 10 000 evaluations"])
    ax.set_title("ZDT1, 10 variables")
    ax.figure.set_size_inches(5, 3.8)
    save(ax.figure, f"algorithm-{name}.png")


ALGORITHMS = {
    "abc": ypea.ABC, "acor": ypea.ACOR, "ba": ypea.BA, "bbo": ypea.BBO, "ca": ypea.CA, "cmaes": ypea.CMAES,
    "de": ypea.DE, "fa": ypea.FA, "ga": ypea.GA, "hs": ypea.HS, "ica": ypea.ICA, "ipop-cmaes": ypea.IPOPCMAES,
    "iwo": ypea.IWO, "jade": ypea.JADE, "lshade": ypea.LSHADE, "nelder-mead": ypea.NelderMead,
    "pso": ypea.PSO, "random-search": ypea.RandomSearch, "sa": ypea.SA, "shade": ypea.SHADE, "tlbo": ypea.TLBO,
}
MULTI = {"nsga2": ypea.NSGAII, "nsga3": ypea.NSGAIII, "moead": ypea.MOEAD, "mopso": ypea.MOPSO, "spea2": ypea.SPEA2}


@figure
def algorithms():
    for name, cls in ALGORITHMS.items():
        _search_snapshots(cls, name)
    for name, cls in MULTI.items():
        _front_snapshot(cls, name)


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    for key in wanted:
        if key.startswith("algorithm-"):
            name = key.removeprefix("algorithm-")
            (_search_snapshots if name in ALGORITHMS else _front_snapshot)({**ALGORITHMS, **MULTI}[name], name)
        else:
            FIGURES[key]()
