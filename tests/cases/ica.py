from ypea.algorithms.ica import ICA

CASES = {
    "ICA": (lambda **k: ICA(**k), 1e-21),
}
