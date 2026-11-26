# Test problems

See the user guide page [Test problems](../guide/test-problems.md) for an overview.

## Engineering design

::: ypea.problems.engineering
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - pressure_vessel
        - welded_beam
        - spring
        - speed_reducer
        - three_bar_truss
        - gear_train

## Traveling salesman

::: ypea.problems.tsp
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - tsp
        - random_tsp
        - circle_tsp
        - plot_tour
        - TourLength
        - TSPLIB_OPTIMA

## Knapsack

::: ypea.problems.knapsack
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - knapsack
        - integer_knapsack
        - matlab_knapsack
        - mknap1
        - KnapsackObjective

## Quadratic assignment and n-Queens

::: ypea.problems.qap
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - qap
        - QAPCost
        - QAPLIB_OPTIMA

::: ypea.problems.nqueens.nqueens
    options:
      heading_level: 3

## Vehicle routing

::: ypea.problems.vrp
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - cvrp
        - cvrp_routes
        - read_cvrplib
        - plot_routes
        - RouteCost

## Generalized assignment

::: ypea.problems.gap
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - gap
        - orlib_gap
        - read_orlib_gap
        - AssignmentCost
        - ORLIB_GAP_OPTIMA

## Hub location

::: ypea.problems.hub
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - hub_location
        - cab
        - read_orlib_cab
        - hub_allocation

## Transportation

::: ypea.problems.transportation
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - transportation
        - ShippingCost

## Instance files

::: ypea.problems.io
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - read_tsplib
        - read_tsplib_tour
        - tsplib_distances
        - read_qaplib
        - read_qaplib_solution
        - read_orlib_mknap
