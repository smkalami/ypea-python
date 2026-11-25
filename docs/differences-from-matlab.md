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
| Reproducibility (F5) | Uses the global random state. | `solve(problem, seed=...)`. The same seed gives the same result. `seed=ypea.StudioRng(n)` draws the same numbers as YPEA Studio. |
| Evaluation | One solution at a time. | Also vectorized (`vectorized=True`) or parallel (`executor=`). Objective values that are not finite (NaN, inf and -inf) count as the worst possible, and NaN constraint values as infinite violation. |
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
- The Metropolis rule follows the problem's comparison rules: with feasibility rules, the difference in violation when the violations differ, otherwise the difference in objective. With a penalty, the difference in penalized objective. The temperature never falls below `final_temp`.
- When only `max_nfe` is set, the cooling schedule is fitted to the number of iterations the budget allows.

### CMA-ES

- Rewritten from Hansen's tutorial (2016). Only the current mean, covariance, paths and step size are kept, so memory no longer grows with `max_iter` (A9).
- `mu` is computed from the population size in effect when the run starts (A8). MATLAB fixed it at construction from the default population size.
- The step size update is the standard one. The extra power of 0.3 in MATLAB is gone (A10).
- `pop_size=None` (the default) gives the standard 4 + floor(3 ln n). MATLAB used 100.
- Samples outside [0, 1] are reflected back at the bounds, and the reflected points are used for the update. MATLAB clipped them, which distorted the distribution (A10).
- The mean is no longer evaluated every iteration. Only the initial mean is evaluated once.
- New settings: `sigma0` (MATLAB fixed it at 0.3), `x0` (initial mean).
- Recombination weights and learning rates follow Table 1 of the tutorial (equations 48 to 58) with positive weights only.

### IWO

- The step size follows the paper's nonlinear schedule from `step_size` to `final_step_size` with `modulation_index`, over the whole run (the share of `max_nfe`, `max_iter` or `max_time` used, whichever is largest). MATLAB multiplied the step size by `step_size_damp = 0.001` every iteration, which stopped the search after one or two iterations (A13).
- `initial_pop_size` is used (A14). MATLAB started with `pop_size` weeds.
- Seed counts follow the paper (linear between the worst and best objective value). When all weeds have the same value, every weed gets the maximum number of seeds (MATLAB divided by zero, A11).
- With constraints, seed counts come from ranks (`fitness="auto"`, the default). `fitness="paper"` or `fitness="rank"` force one rule.

### ABC

- An employed or onlooker bee changes one random component: v_j = x_j + phi (x_j - x_kj), with phi uniform in [-max_acceleration, max_acceleration], as in the paper. MATLAB changed every component.
- A source's trial counter resets to 0 when the source improves. MATLAB never reset it.
- At most one scout per iteration: the source with the largest trial counter, if it has reached the limit. MATLAB replaced every source at the limit.
- Onlooker probabilities use the paper's fitness, 1/(1 + f) for f >= 0 and 1 + |f| otherwise (F2). With constraints, ranks are used (`fitness="auto"`, the default). `fitness="paper"` or `"rank"` force one rule.
- The default number of onlookers is `pop_size` (the paper's SN), and the default trial limit is onlookers times the number of codes, as in Karaboga (2005, Table 2). MATLAB used 100 onlookers and round(0.6 * d * pop_size). Both can be set (`onlooker_count`, `trial_limit`).
- Both bee phases are evaluated as one batch. Each source keeps the best candidate its onlookers found, if it is better.

### ACOR

- Each new solution picks one guiding solution from the archive and samples all its components from it. MATLAB picked a new guide for each component.
- When objective values tie, archive members are kept first.

### BA

- Maximization works (A5). MATLAB crashed on every maximization problem.
- The probabilistic mode scores sites by rank instead of `1 ./ obj_values` (A6). The scores still average 1, and the rejection thresholds (0.6, 0.2, 0.05, 0) are unchanged. This mode follows YPEA for MATLAB.
- In the standard mode, every recruited bee is drawn uniformly from the patch around its site (all components), as in Pham et al. MATLAB changed one random component, which works much better in many dimensions (about 1e-3 against 0.15 on the 10D sphere). `neighborhood="component"` gives the MATLAB move. The probabilistic mode keeps the one component move.
- A site moves to its best recruited bee only when it is strictly better.

### BBO

- The default `alpha` is 1, which copies the immigrating feature as in Simon (2008). `alpha=0.9` gives the MATLAB blend.
- The migration source is never the habitat itself. Populations of one or two habitats no longer divide by zero.
- `keep_rate` and `mutation_prob` must be in [0, 1].

### CA

- Influence types 2, 3 and 4 change only component j with the step for component j (A4). MATLAB added the step to the whole position vector inside the loop over j.
- Survivors are selected from parents and offspring (the best `pop_size` survive), following the select step of the cultural algorithm. MATLAB had no selection, and the default influence type did not converge (about 60 to 70 on the 10D sphere). `selection=False` gives the MATLAB scheme.
- The belief space (normative and situational knowledge) is updated with the problem's comparison rules, so maximization and constraints work.
- At least two solutions are accepted. With one, the normative intervals have zero width and the search stalls.
- The four influence types keep the YPEA for MATLAB definitions.

### FA

- A firefly moves only toward strictly brighter fireflies (A12). The brightest firefly makes a random walk of `alpha * delta * U(-1, 1)`, as in the paper.
- Each firefly moves toward every brighter firefly in turn, and the moves add up. It is evaluated once after all its moves, so an iteration costs `pop_size` evaluations. MATLAB evaluated one candidate per brighter firefly (up to `pop_size` squared evaluations) and kept only the best.

### HS

- The damped bandwidth is used for pitch adjustment, so `fret_width_damp` has an effect (A3).
- Pitch adjustment adds `bandwidth * U(-1, 1)`, as in Lee and Geem (2005). MATLAB used a normal distribution.
- `fret_width_damp=1` and `new_count=1` give the paper's fixed bandwidth and one new harmony per iteration.

### ICA

- Empire power and the imperialistic competition use rank based costs, so they work for maximization, any objective scale, and constraints (F2). The total cost of an empire is c(imperialist) + zeta * mean(c(colonies)), where c is the normalized rank from 0 (best) to 1 (worst).
- A collapsed empire's imperialist really becomes a colony of the winning empire (A7).
- The initial colonies are split in proportion to the imperialists' power, as in the paper. MATLAB drew the empire of each colony at random.
- A colony that revolts is evaluated once. MATLAB evaluated it twice.
- Each iteration evaluates exactly `pop_size` solutions.

### TLBO

- Each phase updates all learners at once (synchronous). MATLAB updated them one by one.
- A learner moves away from its partner only if it is strictly better than the partner, otherwise toward it, as in the paper.
- `pop_size` must be at least 2.
