import ypea

CASES = {
    "GA": (lambda **k: ypea.GA(**k), 0.05),
    "GA-tournament": (lambda **k: ypea.GA(selection="tournament", **k), 0.05),
}
