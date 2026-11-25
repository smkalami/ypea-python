import ypea

CASES = {"NelderMead": (lambda **k: ypea.NelderMead(**k), 1e-15)}
POP_SIZE = {"NelderMead": None}  # the simplex has d + 1 vertices
