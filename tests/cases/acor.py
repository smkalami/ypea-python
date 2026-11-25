from ypea.algorithms.acor import ACOR

CASES = {
    "ACOR": (lambda **k: ACOR(**k), 1e-16),
}
