"""One module per algorithm (or family). Each defines CASES: {name: (factory, sphere_tolerance)}.

``factory(**kw)`` builds the algorithm with the given common settings. The tolerance is the
best objective value it must reach on the 10 dimensional sphere in 20 000 evaluations.
Optional: ``POP_SIZE`` {name: int} overrides the default population size of 40.
"""
