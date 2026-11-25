import ypea

CASES = {"CMA-ES": (lambda **k: ypea.CMAES(**k), 1e-10)}
POP_SIZE = {"CMA-ES": None}  # standard 4 + 3 ln(n)
