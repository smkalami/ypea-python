"""Checks every algorithm must pass (sphere convergence, maximization, every variable type, budget, reproducibility)."""

import importlib
import pickle
import pkgutil

import cases
import numpy as np
import pytest
from problems import neg_sphere_obj

import ypea

CASES, POP_SIZE = {}, {}
for info in pkgutil.iter_modules(cases.__path__):
    mod = importlib.import_module(f"cases.{info.name}")
    CASES.update(mod.CASES)
    POP_SIZE.update(getattr(mod, "POP_SIZE", {}))
NAMES = sorted(CASES)


def make(name, **kw):
    kw.setdefault("pop_size", POP_SIZE.get(name, 40))
    return CASES[name][0](**kw)


@pytest.mark.parametrize("name", NAMES)
def test_sphere_convergence(name, sphere10):
    tol = CASES[name][1]
    r = make(name, max_iter=None, max_nfe=20_000).solve(sphere10, seed=0)
    assert r.best.obj_value < tol
    assert r.nfe == 20_000


@pytest.mark.parametrize("name", NAMES)
def test_maximization(name, sphere10):
    neg = ypea.Problem(sphere10.vars, neg_sphere_obj, sense="max")
    tol = CASES[name][1]
    r = make(name, max_iter=None, max_nfe=20_000).solve(neg, seed=0)
    assert -tol < r.best.obj_value <= 0


@pytest.mark.parametrize("name", NAMES)
def test_every_variable_type(name, mixed):
    r = make(name, max_iter=3).solve(mixed, seed=0)
    sol = r.best.solution
    assert sorted(sol.p) == list(range(5)) and len(sol.q) == 3
    assert np.isclose(sol.fs.sum(), 1) and r.best.obj_value == mixed.objective(sol)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("budget", [1, 7, 123])
def test_budget_not_multiple_of_pop_size(name, sphere10, budget):
    r = make(name, max_iter=None, max_nfe=budget).solve(sphere10, seed=0)
    assert r.nfe == budget


@pytest.mark.parametrize("name", NAMES)
def test_reproducible_and_picklable(name, sphere10):
    alg = make(name, max_iter=5)
    a = alg.solve(sphere10, seed=11)
    b = pickle.loads(pickle.dumps(alg)).solve(sphere10, seed=11)
    assert a.best.obj_value == b.best.obj_value
    assert np.array_equal(a.history.trace_obj_value, b.history.trace_obj_value)


@pytest.mark.parametrize("name", NAMES)
def test_constrained(name, constrained):
    r = make(name, max_iter=None, max_nfe=10_000).solve(constrained, seed=0)
    # optimum is 0.5. HS ends between 0.5 and 0.97 on this problem over seeds 0 to 9
    assert r.best.feasible and r.best.obj_value < (0.7 if name == "HS" else 0.6)
