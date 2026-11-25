from ypea.algorithms.bbo import BBO

CASES = {
    "BBO": (lambda **k: BBO(**k), 1e-6),
}
