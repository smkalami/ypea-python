"""Evaluate the solutions yourself: the ask and tell interface.

The algorithm asks for a batch of solutions, you evaluate them anywhere (a lab, a
cluster job, a simulator), and you tell it the values. Here a function that handles a
whole batch at once stands in for the outside evaluation. Run it as

    python ask_tell.py

It prints the progress, the best design, and checks that solve() gives the same result.
"""

import ypea


def simulate(designs):
    """Stand in for an outside evaluation: the cost of each design, and its constraint.

    Minimize (x - 2)^2 + (y - 1)^2 subject to x + y <= 2 (the optimum is x = 1.5, y = 0.5).
    """
    return [((d.x - 2) ** 2 + (d.y - 1) ** 2, d.x + d.y - 2) for d in designs]


# no objective: the solutions are evaluated outside the toolbox
problem = ypea.Problem([ypea.Real("x", lb=-5, ub=5), ypea.Real("y", lb=-5, ub=5)], objective=None, name="design")

run = ypea.DE(max_nfe=2000, pop_size=20).ask_tell(problem, seed=1)
batches = 0
while not run.done:
    designs = run.ask()  # a list of decoded solutions, and their codes in run.codes
    run.tell(simulate(designs))  # one (f, g) per solution, in the same order
    batches += 1
    if batches % 25 == 0:
        print(f"batch {batches}: {run.algorithm.nfe} evaluations, best f = {run.algorithm.best.obj_value[0]:.6f}")

result = run.result()
print(result.summary())
s = result.best.solution
print(f"best design: x = {s.x:.4f}, y = {s.y:.4f}, f = {result.best.obj_value:.6f}")

# the same run with an objective gives the same result
with_objective = ypea.Problem(problem.vars, lambda d: simulate([d])[0], name="design")
same = ypea.DE(max_nfe=2000, pop_size=20).solve(with_objective, seed=1)
print("same as solve():", same.best.obj_value == result.best.obj_value and same.nfe == result.nfe)
