import ypea

CASES = {
    "SA": (lambda **k: ypea.SA(initial_temp=10, final_temp=1e-4, **k), 1.0),
    "SA-native": (lambda **k: ypea.SA(space="native", initial_temp=10, final_temp=1e-4, **k), 1.0),
}
POP_SIZE = {"SA": 10, "SA-native": 10}
