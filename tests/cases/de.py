import ypea

CASES = {
    "DE/rand/1/bin": (lambda **k: ypea.DE(**k), 0.1),
    "DE/best/2/bin": (lambda **k: ypea.DE("DE/best/2/bin", **k), 0.1),
    "DE/current-to-best/1/exp": (lambda **k: ypea.DE("DE/current-to-best/1/exp", crossover_prob=0.9, **k), 1e-5),
    "DE/rand-to-best/1/bin": (lambda **k: ypea.DE("DE/rand-to-best/1/bin", **k), 1e-3),
}
