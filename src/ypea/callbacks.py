"""Ready made callbacks for ``Algorithm.solve``."""

from __future__ import annotations


class Print:
    """Print progress every ``every`` iterations."""

    def __init__(self, every=1):
        self.every = every

    def on_start(self, alg):
        print(f"{alg.full_name} started on {alg.problem.name}")

    def __call__(self, alg):
        if alg.iter % self.every == 0:
            b = alg.best
            extra = "" if b.V[0] <= 0 else f", violation = {b.V[0]:.3g}"
            print(f"Iteration {alg.iter}: Best Obj. Value = {b.F[0, 0]:.6g}{extra}, NFE = {alg.nfe}")

    def on_end(self, alg, result):
        print(f"Finished: {result.stop_reason}, best = {result.best.obj_value:.6g}")
