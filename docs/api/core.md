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
        - ask_tell
        - params
        - validate
        - initialize
        - iterate
        - request_stop
        - progress
        - planned_iterations
        - result_extra
        - check_problem
        - dominates
        - nondominated_sort
        - crowding_distance

::: ypea.ask_tell.AskTell
    options:
      members:
        - ask
        - tell
        - stop
        - result

::: ypea.result.Result
    options:
      members: true

::: ypea.result.History
    options:
      members: true

::: ypea.result.Front
    options:
      members: true

::: ypea.result.Evaluations
    options:
      members: true

::: ypea.scalarize.scalarize

::: ypea.population.Individual

::: ypea.population.Population
    options:
      members: true

::: ypea.comparison.Comparator
    options:
      members:
        - dominates
        - nondominated_sort
        - crowding_distance

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
