from ypea.algorithms.ica import ICA

CASES = {
    "ICA": (lambda **k: ICA(**k), 1e-21),
}

CASES["ICA-roulette"] = (lambda **k: ICA(competition="roulette", **k), 1e-15)
