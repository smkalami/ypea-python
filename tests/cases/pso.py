import ypea

CASES = {
    "PSO": (lambda **k: ypea.PSO(w=0.5, wdamp=1, c1=1, c2=2, **k), 1e-5),
    "PSO-constriction": (lambda **k: ypea.PSO(**k).use_constriction_coeffs(), 1e-3),
}
