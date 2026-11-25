from ypea.algorithms.ipop_cmaes import IPOPCMAES

CASES = {"IPOP-CMA-ES": (lambda **k: IPOPCMAES(**k), 1e-10)}
POP_SIZE = {"IPOP-CMA-ES": None}  # CMA-ES default for the first run
