import ypea

CASES = {
    "IWO": (lambda **k: ypea.IWO(**k), 1e-2),
    "IWO-rank": (lambda **k: ypea.IWO(fitness="rank", **k), 1e-2),
}
POP_SIZE = {"IWO": 20, "IWO-rank": 20}
