from ypea.algorithms.ba import BA

CASES = {
    "BA": (lambda **k: BA(**k), 2.0),
    "BA-probabilistic": (lambda **k: BA(type="probabilistic", **k), 2e-2),
}

CASES["BA-component"] = (lambda **k: BA(neighborhood="component", **k), 1e-2)
