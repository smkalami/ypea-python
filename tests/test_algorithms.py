"""Checks every algorithm must pass (sphere convergence, maximization, every variable type, budget, reproducibility)."""

import pickle

import numpy as np
import pytest
from problems import neg_sphere_obj

import ypea

# (algorithm, sphere tolerance at 20 000 evaluations in 10 dimensions)
CASES = {
    "PSO": (lambda **k: ypea.PSO(w=0.5, wdamp=1, c1=1, c2=2, **k), 1e-5),
    "PSO-constriction": (lambda **k: ypea.PSO(**k).use_constriction_coeffs(), 1e-3),
    "DE/rand/1/bin": (lambda **k: ypea.DE(**k), 0.1),
    "DE/best/2/bin": (lambda **k: ypea.DE("DE/best/2/bin", **k), 0.1),
    "DE/current-to-best/1/exp": (lambda **k: ypea.DE("DE/current-to-best/1/exp", crossover_prob=0.9, **k), 1e-5),
    "DE/rand-to-best/1/bin": (lambda **k: ypea.DE("DE/rand-to-best/1/bin", **k), 1e-3),
    "GA": (lambda **k: ypea.GA(**k), 0.05),
    "GA-tournament": (lambda **k: ypea.GA(selection="tournament", **k), 0.05),
    "SA": (lambda **k: ypea.SA(initial_temp=10, final_temp=1e-4, **k), 1.0),
}
NAMES = list(CASES)


def make(name, **kw):
    kw.setdefault("pop_size", 10 if name == "SA" else 40)
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
    assert r.best.feasible and r.best.obj_value < 0.6  # optimum is 0.5
