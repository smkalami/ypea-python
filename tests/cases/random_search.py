import ypea

# random sampling: about 40 on the 10D sphere at 20 000 evaluations
CASES = {"RandomSearch": (lambda **k: ypea.RandomSearch(**k), 500.0)}
POP_SIZE = {"RandomSearch": 100}
