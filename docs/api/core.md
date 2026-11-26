# Problems, results and runs

The core classes. All of them are available from the top level package, for example `ypea.Problem`.

::: ypea.problem.Problem
    options:
      members: true

::: ypea.problem.Solution
    options:
      members: false

::: ypea.algorithm.Algorithm
    options:
      members:
        - solve
        - params
        - validate
        - initialize
        - iterate
        - request_stop
        - progress
        - planned_iterations
        - result_extra

::: ypea.result.Result
    options:
      members: true

::: ypea.result.History
    options:
      members: true

::: ypea.population.Individual

::: ypea.population.Population
    options:
      members: true

::: ypea.comparison
    options:
      members:
        - Comparator

::: ypea.callbacks
    options:
      show_root_heading: false
      members:
        - Print

::: ypea.random
    options:
      members:
        - StudioRng
        - derive_seed
