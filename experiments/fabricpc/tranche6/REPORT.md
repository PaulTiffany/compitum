# Tranche 6: trained belief-state FabricPC + Bellman-derived shadow pricing — report

Branch `experiment/fabricpc-trajectory-observer`. FabricPC pin unchanged:
v0.3.2 at `32ae295182ab944b8f084abaf4a40da2c50bab5f` (external checkout
`C:\src\FabricPC`; not vendored, not patched). Nothing in this tranche
affects route selection, `main`, `SwitchCertificate`, or
`constraints.shadow_prices`.

## The question this tranche answers

Tranche 5 tested `fixed random FabricPC transform + trajectory summary +
ridge residual head` and found it inert. It did **not** test `trained
predictive-coding model -> inferred belief -> economically derived
Bellman shadow price` — the actual governing architecture this program
calls for. Per the governing principle: "FabricPC should predict the
future state of scarcity. The Bellman value function should determine
what scarce resources are worth. Do not ask a predictive model to invent
economics that can be calculated exactly." This tranche builds that
architecture properly — an exact online Bellman price oracle first,
genuinely trained FabricPC belief models second — and asks whether
better scarcity prediction can improve pricing at all *before* spending
any compute on training. See docs/adr/0007-belief-state-fabricpc-bellman-pricing.md.

## What was built

- **`src/compitum/regret_lab/belief_regime.py`** — a small, narrow
  hidden-regime environment (one resource, two regimes NORMAL/HIGH,
  `STEPS=10`): exact scalar Bayesian filtering (`filtered_belief`,
  `predict_belief`), reusing `DynamicCase`/`DynamicSequence` unchanged.
  Ground-truth regimes/beliefs returned separately from the sequence,
  never placed in any field a policy can read.
- **`src/compitum/regret_lab/belief_bellman.py`** — `BellmanOracle`: an
  exact, memoized online continuation-value recursion and the discrete
  marginal shadow price (`marginal_price`), hand-verified to spike
  sharply at the exact feasibility boundary and increase monotonically
  with belief in the HIGH regime.
- **`src/compitum/regret_lab/belief_channels.py`** / **`belief_hmm_filter.py`**
  — the declared per-step observation channel, and a generic matrix-based
  HMM filter coded independently of `belief_regime.py`'s scalar formulas,
  cross-validated to agree exactly.
- **`src/compitum/regret_lab/belief_pricing.py`** — `BeliefPricingController`
  plus five interchangeable `BeliefEstimator` implementations (exact,
  HMM, ridge, and a generic precomputed/cached lookup used by both
  FabricPC arms and their shuffled control), so belief-estimation quality
  is the only thing that can differ between arms; `build_belief_training_pairs`
  generates on-policy `(window, next-step belief_prior)` training data
  from an exact-belief reference rollout.
- **`src/compitum/regret_lab/belief_online_optimum.py`** — the true
  online (non-hindsight) Bayes-optimal policy, the principal regret
  comparator this tranche's brief calls for, distinct from both the
  greedy price-based arms and the perfect-foresight hindsight oracle.
  Empirically confirmed (100 seeds) to differ from a naive no-pricing
  policy on 100/100 sequences and to genuinely tighten budget (min
  observed 0.5) — scarcity bites, so pricing has real work to do here.
- **`experiments/fabricpc/tranche6/fabricpc_belief_model.py`** — a real
  trainable FabricPC belief-regression model: one graph topology
  (`source(55)->hidden(16,sigmoid)->belief(1,sigmoid,GaussianEnergy)`,
  `FeedforwardStateInit`), trained two ways from *identical* initial
  parameters — `train_pcn` (genuine local predictive-coding learning)
  and `train_backprop` (ordinary end-to-end backprop, the required
  same-topology control) — with a declared early-stopping rule (fixed
  30-epoch budget, best-validation-MSE snapshot kept). Verified
  end-to-end against the pinned checkout, including a live per-step
  estimator wired through `simulate_policy`.
- **`experiments/fabricpc/tranche6/run_belief_bellman_pilot.py`** — the
  two-phase, gate-ordered pilot: Gate A is checked first using only the
  three arms that require zero training (no pricing, frozen pacing,
  exact belief), before Part B (ridge/HMM/FabricPC training) or the
  remaining five arms ever run.
- 45 new tests across the new `regret_lab` modules (`belief_pricing.py`,
  `belief_online_optimum.py`, plus tranche 6.1-6.3's earlier modules),
  100% line+branch coverage maintained, mypy `--strict` clean, ruff clean.

## Result: stopped at Gate A

**Exact-belief Bellman pricing does not beat frozen pacing.**

| arm | mean regret (vs. exact online optimum) | total deferrals | route-switch rate | high-value rejections |
| --- | --- | --- | --- | --- |
| no pricing | 2.629 | 105 | 68.3% | 0 |
| frozen pacing (baseline) | **1.829** | 54 | 47.9% | 155 |
| exact latent belief + Bellman price | 1.943 | 59 | 73.0% | 139 |

Paired regret delta (arm 3 vs. arm 2, held-out test set, 35 sequences):
mean **+0.114** (arm 3 worse), 95% bootstrap CI **`[-0.229, +0.457]`** —
straddles zero and leans the wrong way. `beats_frozen_pacing: false`.
Violations: zero for both arms (`no_additional_violations: true`, the
one criterion satisfied).

**Robustness check** (not part of the committed artifact, run directly
against five independent test-set seeds before finalizing this report,
since 35 sequences is a small sample): paired deltas were `+0.114`
(seed 4242, the default), `0.0`, `0.0`, `-0.229`, `+0.457` — never
significantly negative on any seed. This is a stable null result, not an
artifact of one unlucky draw.

**Both priced arms crush no pricing** (mean regret 1.83–1.94 vs. 2.63),
confirming pricing matters a great deal in this environment in general —
the null result is specific: the mathematically exact marginal-value
price does not outperform pacing's much simpler budget/time-ratio
heuristic. Two mechanistic signals point at *why*: the exact-belief arm
switches routes far more often (73.0% of steps vs. 47.9%) without a
matching regret improvement, and rejects the immediate-best-utility
option less often than pacing does (139 vs. 155) while still not
converting that into lower regret. This is consistent with the exact
price's own hand-verified shape (Part A: it spikes sharply at the
discrete feasibility boundary rather than varying smoothly) interacting
poorly with the same linear-greedy `price_utilities` decision rule every
arm in this program has used since tranche 3 — a technically correct
price, translated into action through a greedy rule, can still
underperform a smoother heuristic. Tranche 4 found the same general
shape of result ("bad pricing is worse than no pricing").

**Part B (ridge/HMM/FabricPC training) and arms 4–8 were never run.**
Per the ADR's explicit "evaluated in order, stop on failure" instruction,
and per the runtime-discipline mandate, the pilot script checks Gate A
first using only the three training-free arms — when it failed, the
script wrote this report and exited before spending any compute training
FabricPC. This is the intended, cost-disciplined behavior, not a
shortcut: training a predictor to estimate belief would not have been
informative here regardless of how well it learned, since belief quality
is not the bottleneck this environment's Gate A identifies.

## Interpretation, per the ADR's own framework

```text
exact belief does not help -> economics/environment bottleneck
```

More precisely: the bottleneck is the **price-to-action translation**,
not the price computation itself (which is exactly correct by
construction and hand-verified) and not the environment's economics (which
clearly reward pricing over no-pricing at all). A future tranche revisiting
this environment should look at the decision rule (e.g. a smoothed or
hysteresis-aware translation of the exact Bellman price, closer to
`PacingController`'s own deadband/relaxation machinery) rather than at
better belief estimation, which is what tranche 6's own Part B/C
infrastructure was built to test and would not have moved past this gate.

## Honest methodology notes

- 50 train / 15 val / 35 test sequences, one fixed environment
  parameterization (no scenario sweep), matching the "keep this tranche
  intentionally narrow" directive. Gate A's outcome does not depend on
  the train/val split size or FabricPC's epoch budget at all, since
  those only affect Part B, which never ran.
- The exact-belief arm (arm 3) and every other price-based arm in this
  program route via the same greedy `price_utilities` rule
  (`priced_utility[m] = utility[m] - lambda*consumption[m]`, argmax over
  feasible models) — deliberately, so belief-estimation quality is the
  only thing that could differ between arms. This means Gate A's failure
  is a statement about that decision rule's interaction with an exactly
  -correct price in this environment, not about the price's correctness
  or the environment's economic structure in isolation.
- Three `PolicyRunResult` fields (`violation_count`/`violation_magnitude`,
  `depleted_budget_events`, `high_value_rejections` in the sense of "an
  affordable, available, strictly-higher-utility option existed and was
  never worth deferring for") were found, during test-writing for
  `belief_online_optimum.py`, to be provably always zero for the true
  online-optimal policy given this environment's own parameters (exact
  feasibility-consistent action selection; `GRID_UNIT`/replenishment
  integer arithmetic; "opportunity"'s flat, non-varying utility) — the
  corresponding dead defensive branches were removed rather than left
  untested, matching this program's established practice.
- Part B/C's infrastructure (`fabricpc_belief_model.py`,
  `belief_pricing.py`'s ridge/HMM/lookup estimators) is built, smoke
  -verified end-to-end against the real pinned FabricPC checkout, and
  committed as reusable, non-dead code — not exercised by this pilot's
  final run because Gate A stopped it first.

## Preserved conclusions

Tranches 1–5's results stand unchanged. Tranche 5's supported conclusion
remains narrow ("fixed, untrained FabricPC trajectory features used
through a bounded ridge residual did not improve regret beyond frozen
pacing") and is not generalized by this tranche's own negative result,
which is about a *different* mechanism (exact-belief pricing's
interaction with greedy routing, not FabricPC prediction quality at
all — Part B/C were never reached).
