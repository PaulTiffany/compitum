# ADR 0003: dynamic route-specific constraints and regret (tranche 3)

Status: accepted, observation-only. Supersedes nothing in ADR 0001/0002;
follows directly from ADR 0002's tranche-2-outcome addendum and explicit
user direction after reviewing tranche 2's negative result.

## Governing correction

FabricPC was never meant merely to predict a static critical-relaxation
label. It was meant to help Compitum anticipate the opportunity cost of
scarce resources across a sequence of decisions. Tranche 2 showed that
question cannot be answered under the frozen constraint representation
(shared `xB`, `np.all` feasibility across models) because that
representation does not ordinarily create a route-specific feasible set —
its oracle target is close to a deterministic function of present slack,
which a static predictor already has. Tranche 3 must therefore test the
claim in the unit that actually matters — **held-out cumulative constrained
regret** in an environment where different model choices genuinely consume
different, cumulative, time-varying resources — not classification accuracy
or critical-relaxation MAE against a present-slack label.

## Regret is the primary metric, not a byproduct

```text
Reduce cumulative constrained routing regret by improving the timeliness
and usefulness of Compitum's constraint prices, without increasing
violations, unnecessary deferrals, or unacceptable latency.
```

Classification accuracy, recall, and MAE (tranche 2's metrics) remain useful
diagnostics but are not, by themselves, an activation criterion. A pressure
-prediction error matters in proportion to its downstream decision cost: a
model can lose on unweighted accuracy while correctly identifying the few
constraint events that dominate regret, and a shuffled control can win on
MAE while inducing worse routing decisions. Tranche 3's gate is
decision-focused: it is evaluated on simulated regret, not on any
intermediate prediction score.

## Preserved semantics (unchanged from ADR 0001/0002)

`SwitchCertificate`, `constraints.shadow_prices`,
`ReflectiveConstraintSolver`, routing decisions, utility calculations,
controller behavior, the `v0.2.0` tag, public documentation. Nothing in
this tranche modifies the frozen solver or production routing behavior.

## New naming discipline for tranche 3

Everything below lives in `src/compitum/regret_lab/`, a package name chosen
specifically to read as an experimental laboratory, not a production
control path. Nothing here is named `shadow_price`. The online pricing
variable introduced here is called `dual_price` (a `DualController`'s
`lambda_`), kept in a dataclass separate from, and never assigned into,
`constraints.shadow_prices`. Three distinct things must never be conflated:

```text
existing shadow_prices     = frozen, retrospective, fixed-1e-5 finite
                              -difference diagnostic (v0.2.0, production)
regret_lab.dual_price      = experimental, online, primal-dual reference
                              controller state (this tranche, experiment
                              -only, never wired into production)
FabricPC pressure forecast = an input signal informing dual_price updates
                              in one comparison arm, never a replacement
                              for either of the above
```

## Experiment-owned dynamic resource model

`src/compitum/regret_lab/environment.py` declares route-specific,
time-varying resource consumption:

```text
consumption[t, model, resource]
remaining[t + 1] = remaining[t] - realized_consumption[t, selected_model]
                                 + replenishment[t]
```

Resources modeled: an abstract budget-like quantity and a quota-like
quantity (two interacting resources are enough to exercise multi-constraint
interaction without combinatorial blowup in the pilot). Each
`DynamicCase` (one routing decision) carries, per model: `base_utility`
(ignoring resource cost), `expected_consumption` (the forecast available at
decision time — may differ from ground truth), and `realized_consumption`
(ground truth, fixed at generation time so the hindsight oracle and every
online policy are scored against the exact same realized outcomes). A
`revelation_delay` models delayed realized outcomes: the online simulator
reserves budget using `expected_consumption` at decision time and corrects
the ledger by `realized - expected` once the delay elapses — a genuine
source of policy regret from forecast error, not merely from the resource
constraint itself.

Eight controlled scenarios cover the brief's required cases: permanently
slack controls, a single scarce-resource period, a demand burst, conserving
now to enable a much better future route, premature conservation causing
unnecessary regret, multiple interacting budgets, stochastic forecast
error, and delayed realized outcomes. Each is deterministic given a seed and
independently hand-checkable.

## Hindsight constrained oracle

`src/compitum/regret_lab/hindsight.py` computes the exact constrained
sequence optimum via memoized search over `(step, discretized remaining
-budget state)`: at each step, choose among feasible models (or defer) to
maximize cumulative *realized* utility subject to the cumulative resource
ledger, using perfect foresight of realized consumption (the oracle is not
subject to `expected_consumption` forecast error or revelation delay — it
has ground truth throughout). Budgets are generated on a fixed rational
grid and represented as scaled integers internally so the search is exact
(no floating-point drift), matching the project's existing preference for
exact closed-form or exactly-verified oracles over heuristics. If the
reachable-state count exceeds a documented cap (guarding against
combinatorial blowup on longer sequences than the pilot uses), the oracle
falls back to a greedy policy and reports a conservative optimality gap
against the trivial per-step unconstrained-utility upper bound, rather than
silently returning a possibly-loose number as if it were exact.

## Non-FabricPC online primal-dual baseline

`src/compitum/regret_lab/dual_controller.py` implements an experiment-only
reference controller, established *before* FabricPC is tested against it —
per the user's explicit instruction that FabricPC needs a valid dual
baseline to improve upon, not a strawman:

```text
lambda[t+1] = clip(lambda[t] + eta * utilization_error[t], 0, lambda_max)
priced_utility[t, model] = base_utility[t, model]
                            - dot(lambda[t], expected_consumption[t, model])
```

This is not claimed to be optimal or production-ready; it is the minimum
coherent baseline FabricPC must beat. A separate, dependency-free
exponentially-weighted-moving-average consumption forecaster
(`regret_lab.forecaster`) stands in for "a simple non-FabricPC sequential
model" (arm 3) — FabricPC (arm 4) must beat this too, not just the naive
arm 2.

## Comparator and regret accounting

Primary metric: `cumulative constrained regret = hindsight optimum utility
- policy cumulative realized utility`, reported separately from feasibility.
Violations (count and cumulative magnitude) are never folded into the
regret scalar as a penalty — a policy that "solves" regret by silently
violating budgets must show that as a separate, visible number. Additional
reported metrics: utility per resource unit, avoidable/unnecessary
deferrals, route-switch rate, depleted-budget events, tail regret (worst
-quantile per-sequence regret), p50/p95 decision latency, observer
failures/refusals.

## Five paired arms and the real gate

1. static/frozen pricing (no dual controller: greedy max `base_utility`
   among expected-feasible models);
2. online primal-dual, no learned predictor (raw `expected_consumption`
   drives pricing);
3. online primal-dual + non-FabricPC EWMA sequential predictor;
4. online primal-dual + FabricPC trajectory-pressure prediction
   (`experiments/fabricpc/tranche3/`, JAX-side, dependency-isolated exactly
   as tranches 1-2);
5. arm 4 with shuffled/sequence-mismatched trajectories (negative control).

Paired: identical sequences, seeds, realized outcomes, and initial budgets
across all five arms.

**Gate:** on untouched held-out sequences, arm 4 must reduce cumulative
constrained regret relative to *both* arm 2 and arm 3, must not increase
hard violations, and must be distinguishable from arm 5. Gated on effect
size and paired uncertainty (e.g. bootstrap CI over paired per-sequence
regret differences), not on raw classification accuracy or MAE.

## Observation-only status, unchanged

No production route is changed; no existing certificate field is changed;
no `v0.2.0` source semantics are rewritten; no persistent dual is introduced
into public Compitum; no claim is made that FabricPC helps until this gate
passes. The simulated policies exist purely to compute counterfactual
regret in `regret_lab`'s own offline environment — they never feed back
into live Compitum routing.

## Outcome (2026-07-23)

The bounded pilot (`experiments/fabricpc/tranche3/`) ran the pre-registered
five-arm comparison. The gate failed: arm 4 (dual + FabricPC) was
significantly *worse* than arm 2 (dual, no predictor; 95% CI on the paired
regret delta `[0.159, 1.282]`, entirely positive), statistically
indistinguishable from arm 3 (EWMA), and statistically indistinguishable
from its own shuffled-trajectory control (arm 5). Full detail, including a
second, FabricPC-independent finding (both forecast-correction mechanisms
tested trade higher regret for fewer depleted-budget events -- a genuine
instance of the `premature_conservation_regret` failure mode showing up in
aggregate), is in `experiments/fabricpc/tranche3/REPORT.md`. This falsifies
the tranche 3 hypothesis as stated in the governing correction above, under
the specific dual-controller/forecaster/graph design tested here. Per the
stop boundary, no tranche 4 activation mechanism is introduced and no
claim of improvement is made.

## Stop boundary for this tranche

Complete only: this ADR; a tested offline sequence environment; an exact
(or bounded-quality, gap-reported) hindsight oracle; the non-FabricPC dual
baseline and EWMA predictor; the five paired arms; a bounded pilot; an
honest, regret-centered report. Stop before push, PR, production
integration, stable schema changes, public documentation changes, or any
claim of improvement.
