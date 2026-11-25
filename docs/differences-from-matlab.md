# Differences from YPEA for MATLAB

YPEA for Python keeps the ideas and names of YPEA for MATLAB (v1.1), but it fixes known defects and follows the published versions of the algorithms. Results will not match MATLAB run for run, because the random number generators differ anyway. This page lists every behavior change. It is also the to-do list for a future update of the MATLAB toolbox.

Item codes (F1, V1, A1, and so on) label the defects found in a review of the MATLAB code (F for the framework, V for variable types, A for algorithms).

## Problems and variables

| Change | MATLAB | Python |
|---|---|---|
| Indices | Permutations, selections and partitions are 1 based. | 0 based. |
| Integer decoding (V1) | A code of exactly 1 decodes to `ub + 1`. | Decodes to `ub`. Every value in [lb, ub] is equally likely. |
| Fixed sum decoding (V2) | All zero codes give NaN. | The target is shared equally. |
| Fixed sum types | `fixed_sum`, `fixed_row_sum`, `fixed_column_sum`. | One type, `FixedSum(axis=None, 1 or 0)`. |
| Encoding | Decode only. | Every type also has `encode()`, so known solutions can be turned into codes. |
| Variable type lookup (F7) | Scans the toolbox folder on every `ypea_var` call. | Types are classes. Users subclass `Variable` to add a type. |
| Stopping rules | `goal` and `max_nfe` are set on the problem, and `max_iter` (100 by default) always applies. | `target`, `max_nfe`, `max_iter` and `max_time` are set on the algorithm. The run stops at whichever limit comes first. `max_iter` defaults to None, and 100 iterations apply only when no limit is given at all, so `max_nfe` alone is never cut short. |
| Constraints | Users write a penalty into the objective. | The objective may return `(f, g)`. `Problem(constraints="feasibility")` (default) or `constraints="penalty"`. |

## Algorithm framework

| Change | MATLAB | Python |
|---|---|---|
| Ties (F1) | A tie counts as "better". | Strictly better only. DE still accepts ties, as in the paper. |
| Evaluation budget (F3) | `max_nfe` is checked after each iteration, so it can be exceeded. | Never exceeded. Rows past the budget are not evaluated. |
| Best solution | Each algorithm updates it, some only once per iteration. | Updated after every evaluation by the framework. The exact trace of improvements is kept. |
| Selection probabilities (F2) | `exp(-beta * f)` on raw objective values. Overflows, and depends on the scale and sign of `f`. | Based on ranks, so independent of scale and sign. |
| Reproducibility (F5) | Uses the global random state. | `solve(problem, seed=...)`. The same seed gives the same result. |
| Evaluation | One solution at a time. | Also vectorized (`vectorized=True`) or parallel (`executor=`). NaN objective or constraint values count as the worst possible. |
| Display | On by default, one line per iteration. | Off by default. Use `display=True` or `callbacks=[ypea.callbacks.Print()]`. |
| Plots | Methods of the algorithm class. | `ypea.plot.convergence(result)`. |

## Algorithms

### PSO

- Same update rule and defaults. All particles move at once (synchronous). In MATLAB the global best changed during the loop over particles.

### DE

- `rand-to-best` works (A1, it crashed on a typo in MATLAB).
- Exponential crossover follows Storn and Price (1997): from a random start, a run of consecutive components whose length depends on `crossover_prob` (A2). MATLAB ignored `crossover_prob` and could copy more than all components.
- `target-to-best` is also available under its usual name, `current-to-best`.
- A strategy is given as `DE("DE/best/2/bin")` or with the separate settings.
- Trial vectors are built from the population at the start of the iteration (synchronous).

### GA

- Roulette wheel selection is rank based (see F2). `selection="tournament"` is new.

### SA

- Each solution keeps the best of its own `move_count` moves and compares it with itself. In MATLAB the k-th best move of the whole population was compared with the k-th solution, which mixed unrelated solutions.
- With constraints, the Metropolis rule uses the difference in violation when a solution is infeasible.
- When only `max_nfe` is set, the cooling schedule is fitted to the number of iterations the budget allows.

The other algorithms are not ported yet.
