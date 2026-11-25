from ypea.algorithms.ca import CA

# influence types 1, 2 and 4 are weak on the sphere, so their tolerances are loose
CASES = {
    "CA": (lambda **k: CA(**k), 1e-6),
    "CA-1": (lambda **k: CA(influence_type=1, **k), 1e3),
    "CA-2": (lambda **k: CA(influence_type=2, **k), 20),
    "CA-4": (lambda **k: CA(influence_type=4, **k), 100),
}

CASES["CA-no-selection"] = (lambda **k: CA(selection=False, **k), 200.0)  # MATLAB scheme, does not converge
