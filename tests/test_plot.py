import numpy as np
import pytest
from problems import sphere_obj

import ypea
from ypea.experiment import Experiment

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


def test_band_boxes_and_ranks():
    x = np.arange(1, 11)
    y = np.array([10.0 ** -x, 2 * 10.0 ** -x, 3 * 10.0 ** -x])
    ax = ypea.plot.band(x, {"A": y, "B": 2 * y})
    assert len(ax.lines) == 2 and ax.get_yscale() == "log" and len(ax.collections) == 2
    assert np.allclose(ax.lines[0].get_ydata(), np.median(y, axis=0))
    ax = ypea.plot.band(x, {"A": y}, center="mean", log=False)
    assert np.allclose(ax.lines[0].get_ydata(), y.mean(axis=0)) and ax.get_yscale() == "linear"
    with pytest.raises(ValueError):
        ypea.plot.band(x, {"A": y}, center="max")
    ax = ypea.plot.boxes({"A": [0.0, 1.0, 2.0], "B": [1.0, 2.0, 3.0]})
    assert ax.get_yscale() == "linear" and [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
    ax = ypea.plot.rank_bars({"A": 1.5, "B": 1.5})
    assert len(ax.patches) == 2
    plt.close("all")


def test_report_figures(tmp_path):
    p = ypea.Problem(ypea.Real("x", shape=2, lb=-5, ub=5), sphere_obj, name="sphere2", optimum=0.0)
    exp = Experiment({"PSO": ypea.PSO(pop_size=8), "DE": ypea.DE(pop_size=8)}, [p], runs=2, max_nfe=80)
    rep = exp.run().report()
    ax = rep.plot_convergence("sphere2")
    assert len(ax.lines) == 2 and ax.get_title() == "sphere2"
    assert ax.lines[0].get_xdata()[-1] == 80
    for ext in ("png", "pdf", "svg"):
        ax.figure.savefig(tmp_path / f"conv.{ext}")
        assert (tmp_path / f"conv.{ext}").stat().st_size > 0
    assert len(rep.plot_box("sphere2").get_xticklabels()) == 2
    assert len(rep.plot_ranks().patches) == 2
    assert rep.plot_convergence("sphere2", metric="obj_value").get_ylabel() == "Best objective value"
    plt.close("all")


def test_front_plots():
    from ypea.benchmarks.multi import pareto_front

    true2 = pareto_front("zdt1", 50)
    ax = ypea.plot.front(true2[::5] + 0.1, true2[::7] + 0.2, true_front=true2, labels=["A", "B"])
    assert len(ax.collections) == 3 and [t.get_text() for t in ax.get_legend().get_texts()] == ["True front", "A", "B"]
    true3 = pareto_front("dtlz2", 30, m=3)
    ax = ypea.plot.front(true3, objectives=["cost", "weight", "risk"])
    assert ax.name == "3d" and ax.get_zlabel() == "risk"
    with pytest.raises(ValueError, match="3d"):
        ypea.plot.front(true3, ax=plt.figure().gca())
    true5 = pareto_front("dtlz2", 30, m=5)
    ax = ypea.plot.front(true5[:4], true_front=true5)
    assert len(ax.lines) == 4 + len(true5) and len(ax.get_xticks()) == 5
    ax = ypea.plot.front(true2, kind="parallel")
    assert len(ax.lines) == len(true2)
    with pytest.raises(ValueError):
        ypea.plot.front(true5, kind="scatter")
    with pytest.raises(ValueError):
        ypea.plot.front(true2, true3)
    with pytest.raises(ValueError):
        ypea.plot.front()
    plt.close("all")
