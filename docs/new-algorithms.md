# Algorithms not in YPEA for MATLAB

These four algorithms are strong modern baselines for benchmark comparisons, such as the CEC suites. Like the other algorithms, they work with every problem and variable type.

## Adaptive Differential Evolution: JADE, SHADE, L-SHADE

All three use the DE/current-to-pbest/1/bin strategy with an external archive of replaced parents. They adapt the scale factor F and the crossover rate CR from successful trial vectors.

- F ~ Cauchy(m, 0.1), drawn again while F <= 0 and truncated to 1. CR ~ N(m, 0.1), clipped to [0, 1].
- A trial vector replaces its parent when it is at least as good. It counts as a success only when it is strictly better.
- A trial component outside the bounds is set to the midpoint between the parent's value and the violated bound.

### JADE

`ypea.JADE(pop_size=100, p_best_rate=0.05, c=0.1, archive=True)`

mu_CR and mu_F start at 0.5. After each iteration, mu_CR = (1 - c) mu_CR + c mean(S_CR), and mu_F = (1 - c) mu_F + c lehmer_mean(S_F).

Reference: J. Zhang and A. C. Sanderson, "JADE: Adaptive differential evolution with optional external archive", IEEE Transactions on Evolutionary Computation 13(5), 945 to 958, 2009.

### SHADE

`ypea.SHADE(pop_size=100, memory_size=100, archive_rate=1.0, p_best_max=0.2)`

A success history memory of H slots for F and CR, starting at 0.5. Each trial vector uses a random slot and its own pbest rate, drawn from [2/N, 0.2]. After each iteration, one slot (in turn) gets the weighted Lehmer mean of the successful F values and the weighted mean of the successful CR values, weighted by the improvements.

Reference: R. Tanabe and A. Fukunaga, "Success-history based parameter adaptation for differential evolution", IEEE CEC 2013, 71 to 78.

### L-SHADE

`ypea.LSHADE(pop_size=None, min_pop_size=4, memory_size=6, archive_rate=2.6, p_best_rate=0.11)`

SHADE with a fixed pbest rate, the weighted Lehmer mean and a terminal value for CR, and linear population size reduction. After each iteration, the size becomes round(N_init + (N_min - N_init) * nfe / max_nfe), and the worst members are removed. N_init defaults to 18 D, where D is the number of codes. Without `max_nfe`, the reduction follows the planned number of iterations.

Reference: R. Tanabe and A. Fukunaga, "Improving the search performance of SHADE using linear population size reduction", IEEE CEC 2014, 1658 to 1665.

## IPOP-CMA-ES

`ypea.IPOPCMAES(pop_size=None, sigma0=0.5, incpopsize=2, max_restarts=None)`

Runs CMA-ES and restarts it from a new random mean when a run stops making progress, multiplying the population size by `incpopsize` each time. The result is the best solution over all runs.

| Setting | Default | Meaning |
|---|---|---|
| `pop_size` | None | population size of the first run (None gives 4 + floor(3 ln n)) |
| `sigma0` | 0.5 | initial step size of every run, in code units (half the search space, as in the paper) |
| `x0` | None | initial mean of the first run |
| `incpopsize` | 2 | factor applied to the population size at each restart |
| `max_restarts` | None | no limit. When reached, the run stops with the reason "max_restarts" |
| `tol_fun` | 1e-12 | range of recent objective values (0 turns it off) |
| `tol_hist_fun` | 1e-12 | range of the best values over the last 10 + ceil(30 n / lambda) iterations (the paper says zero, a tiny tolerance avoids depending on exact ties) |
| `tol_x` | 1e-12 * sigma0 | smallest step size in every coordinate |
| `max_condition` | 1e14 | condition number of the covariance matrix |

After a run, `restarts`, `lam_current` and `restart_log` (a list of evaluations, new population size and reason) describe the restarts.

References: A. Auger and N. Hansen, "A restart CMA evolution strategy with increasing population size", IEEE CEC 2005, 1769 to 1776. N. Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772, 2016.
