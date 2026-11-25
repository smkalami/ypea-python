from ypea.algorithms.adaptive_de import JADE, LSHADE, SHADE

# tolerances: about 10 to 50 times the worst best value of seeds 0, 1, 2 (pop_size 40, L-SHADE 18 D)
CASES = {
    "JADE": (lambda **k: JADE(**k), 1e-23),
    "SHADE": (lambda **k: SHADE(**k), 1e-27),
    "L-SHADE": (lambda **k: LSHADE(**k), 1e-17),
}
POP_SIZE = {"L-SHADE": None}  # 18 D
