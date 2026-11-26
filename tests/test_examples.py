"""The scripts in examples/ run, and every Python code block of the documentation runs."""

import re
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# experiments and tuning take minutes, so they run only with the slow tests (CI)
SLOW = {"experiment_cec2017", "experiments.md", "tuning.md"}


def _mark(paths, key):
    return [pytest.param(p, marks=pytest.mark.slow) if key(p) in SLOW else p for p in paths]


EXAMPLES = _mark(sorted((ROOT / "examples").glob("*.py")), lambda p: p.stem)
# differences-from-matlab.md has no code
SKIP_PAGES = {"differences-from-matlab.md"}
PAGES = _mark(sorted(p for p in (ROOT / "docs").rglob("*.md") if p.name not in SKIP_PAGES), lambda p: p.name)
FENCE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


@pytest.fixture
def agg(monkeypatch, tmp_path):
    """Draw figures off screen and work in a temporary folder."""
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    if matplotlib is not None:
        matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    yield
    if matplotlib is not None:
        import matplotlib.pyplot as plt

        plt.close("all")


@pytest.mark.skipif(not EXAMPLES, reason="the examples folder is not part of this copy")
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example(path, agg, tmp_path, monkeypatch, capsys):
    # sphere_all.py takes algorithm names (the algorithm pages of the docs run every algorithm),
    # the others an output folder
    args = ["PSO", "DE"] if path.stem == "sphere_all" else [str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", [str(path), *args])
    try:
        namespace = runpy.run_path(str(path), run_name="__main__")
    except SystemExit as e:  # an example without matplotlib stops after its text output
        assert e.code in (0, None)
        namespace = {}
    assert capsys.readouterr().out
    for make in namespace.get("ALGORITHMS", {}).values():
        make()  # every setting of sphere_all.py is valid


@pytest.mark.skipif(not PAGES, reason="the docs folder is not part of this copy")
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(ROOT / "docs").as_posix())
def test_docs_code_blocks(page, agg):
    blocks = FENCE.findall(page.read_text(encoding="utf-8"))
    if not blocks:
        pytest.skip("no Python code")
    if "matplotlib" in "".join(blocks) or "plot" in "".join(blocks):
        pytest.importorskip("matplotlib")
    namespace = {"__name__": "__main__"}
    for i, code in enumerate(blocks):
        exec(compile(code, f"{page.name}, block {i + 1}", "exec"), namespace)
