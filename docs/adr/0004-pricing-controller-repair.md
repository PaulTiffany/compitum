# ADR 0004: pricing-controller repair (tranche 4)

Status: accepted, observation-only. Supersedes nothing in ADR 0001/0002/0003;
follows directly from tranche 3's outcome and explicit user direction.

## The finding that reorders the roadmap

Tranche 3's pilot showed, on held-out sequences (mean cumulative regret):

```text
static / no pricing            2.778
online dual, no predictor      2.982   (+0.204 vs no pricing)
online dual + EWMA             3.671   (+0.893 vs no pricing)
online dual + FabricPC         3.669   (+0.891 vs no pricing)
```

**Bad pricing is worse than no pricing.** The reactive dual controller
already loses to doing nothing; both forecast-correction mechanisms tested
on top of it lose by even more. Depleted-budget events and deferrals fell
sharply under the learned arms (10 -> 0-3, 8 -> 2-6) while regret rose --
the policies preserved resources more aggressively without allocating them
better. This is premature conservation, not good pricing.

**Consequence for the roadmap:** asking FabricPC (or any learned
predictor) to improve a pricing mechanism that is *itself* net-negative
risks training that predictor merely to cancel a defective controller --
not to add genuine information. Until a non-learned pricing policy beats
no pricing on regret, adding intelligence to the price signal is
premature. This tranche is exclusively about establishing that baseline.
No FabricPC work is done here except passive verification that existing
infrastructure still imports and runs.

## The distinction this tranche is built around

```text
resource preservation  !=  good resource allocation
```

Finishing a sequence with unused budget is not success if the policy
rejected high-value routes that the budget existed to fund. Every
controller here is evaluated on regret against the exact hindsight
optimum, with terminal unused resources and rejected-high-value-route
counts reported as separate diagnostics precisely so hoarding cannot hide
behind a superficially "safe-looking" outcome (few violations, low
depletion).

## New primary gate

```text
R(pricing baseline) < R(no pricing), paired, bootstrap-CI-significant,
with no increase in hard violations.
```

Until a candidate passes this gate, no learned observer (EWMA, FabricPC,
or anything else) is layered on top of it. The later, stricter sequence
for any future FabricPC work is:

```text
1. establish a non-learned pricing controller that beats no pricing
2. test whether a simple non-FabricPC predictor improves it
3. only then test whether FabricPC improves both
```

FabricPC's eventual gate must beat: no pricing, the repaired non-learned
controller, a simple non-FabricPC sequential predictor, AND its own
shuffled-trajectory control -- strictly more arms than tranche 3's gate,
not fewer.

## Pricing-controller interface

`src/compitum/regret_lab/pricing.py` introduces a small, uniform interface
so `simulate_policy` can drive any non-learned controller identically:
a `lambda_price: Dict[str, float]` attribute (fed into the existing,
unchanged `price_utilities`) and an `update(context: PricingUpdateContext)`
method, where `PricingUpdateContext` carries the raw per-step ingredients
(reservation, remaining before/after, step, total_steps) every controller
variant needs -- not a single pre-derived error number, since the pacing
-family controllers need cumulative history the old single-step formula
discarded.

`simulate_policy`'s pricing hook is renamed `dual_controller` ->
`pricing_controller` and generalized to this interface. **The existing
`DualController` class (`dual_controller.py`) is unchanged byte-for-byte**
-- its fields, its own `update(utilization_error)` signature, and
`price_utilities` are untouched. A new `ReactiveController` adapter wraps
it, reproducing tranche 3's exact utilization_error formula
(`reservation[r] - remaining_after[r] / steps_left`) so that arm's
behavior is verified identical to tranche 3's, not merely similar (see the
regression check in the tranche 4 pilot report).

`experiments/fabricpc/tranche2/run_pilot.py` and
`experiments/fabricpc/tranche3/run_pilot.py` are historical, already
-executed artifacts (like tranche 1's); they are not re-run against this
interface change, matching how each tranche's pilot script has always been
a point-in-time snapshot, not living infrastructure.

## Pacing error formulation

Given the brief's two candidate formulations, this tranche uses the
cumulative-usage-vs-horizon-trajectory form, since it requires no demand
forecasting (staying strictly non-learned):

```text
e_t,i = (cumulative_usage_t,i / max(t+1, 1)) - (total_available_i / T)
```

where `cumulative_usage_t,i` is the controller's own running sum of what
it has *reserved* for resource `i` through step `t` (the controller's own
real-time information -- it does not wait for delayed revelation), and
`total_available_i = initial_budget_i + sum of every step's declared
replenishment for the whole sequence`.

**Modeling assumption, stated explicitly:** computing `total_available_i`
requires knowing the sequence's full replenishment schedule in advance.
This is treated as *known service capacity* (e.g. "this quota resets on a
fixed schedule"), not a forecast of future demand or utility outcomes --
no consumption, utility, or routing-choice information is used ahead of
when it actually occurs. This is the least assumption-heavy way to make
pacing meaningful and is declared here rather than left implicit.

One flexible `PacingController` class realizes four of the brief's six
candidates via parameters, rather than four near-duplicate classes:

```text
plain pacing            deadband=0,  relax_gamma=1,  no smoothing, no step bound
pacing + hysteresis      deadband=delta>0, relax_gamma=1
asymmetric               deadband small/0, relax_gamma>1 (faster relaxation
                         after over-conservation), rise_scale<=1 (slower rise)
bounded + smoothed       ema_alpha<1 (smooths the error before applying h),
                         max_step bounds |lambda change| per step, explicit lambda_max
```

using the update rule `lambda_{t+1,i} = clip(lambda_t,i + eta * h(e_t,i), 0,
lambda_max)` with `h` the piecewise deadband/asymmetric-relaxation function
from the brief. The remaining two candidates ("no pricing", "existing
reactive controller") are the immutable control and the untouched
tranche-3 baseline respectively.

## Diagnostics for premature conservation

`PolicyRunResult` gains `terminal_remaining` (unused resources at sequence
end) and `high_value_rejections` (steps where a genuinely-affordable,
higher-base-utility model was available under realized consumption but a
lower-utility model was chosen because of pricing -- the direct,
per-step signature of hoarding). A separate, explicitly-labeled *heuristic*
diagnostic splits per-step regret (policy's step utility vs. the hindsight
oracle's own per-step choice at that same step) into "attributable to
conservation" (policy underperformed while resources were not genuinely
scarce) vs. "attributable to depletion" (policy underperformed while
genuinely resource-constrained). This is a per-step attribution, not a
rigorous decomposition of the exact end-to-end regret (which is a joint,
path-dependent quantity) -- reported as a diagnostic to distinguish
hoarding from genuine scarcity, not as a replacement for the primary
regret metric.

## Parameter discipline

Pacing-family parameters (eta, deadband, relax_gamma, rise_scale,
ema_alpha, max_step) are selected via a small, declared bounded grid search
on the TRAINING sequences only (the same sequences tranche 3 designated
train, never touched by held-out evaluation), scored by mean regret
subject to zero increase in hard violations. The winning configuration per
controller family is frozen before any held-out test-sequence evaluation
and recorded verbatim in the pilot report.

## Outcome (2026-07-23)

The bounded pilot (`experiments/fabricpc/tranche4/`) ran the pre-registered
six-arm comparison. Full detail, including a genuine bug found and fixed
along the way (the dataset generator's RNG seeding used Python's
process-randomized `hash()`, silently breaking cross-process
reproducibility since tranche 3), is in
`experiments/fabricpc/tranche4/REPORT.md`. Summary:

- The reactive controller (tranche 3's failed reference, parameters
  unchanged) reproduces the "bad pricing is worse than no pricing" finding
  again under the corrected, now-genuinely-reproducible dataset (paired
  regret delta +0.341, 95% CI `[0.065, 0.648]`, entirely positive).
- All four repaired pacing-family controllers, once a too-narrow initial
  parameter grid was caught and widened (found empirically, not assumed),
  achieve a large mean regret improvement (-2.5) -- but it is concentrated
  entirely in 2 of 16 held-out sequences (both `conserve_enables_better_future`),
  with **exactly zero** behavioral difference from no pricing on the other
  14 sequences across the remaining 7 scenarios. The pre-registered
  bootstrap-CI gate correctly declines to certify this as a general
  improvement (the upper CI bound sits exactly at 0.0, reflecting how
  concentrated -- not broadly distributed -- the effect is).
- A clean illustration of resource-preservation-!=-good-allocation fell
  out directly: pacing and reactive both reject a genuinely-affordable,
  higher-utility model 8 times each (`high_value_rejections`), but
  pacing's rejections are net beneficial (regret improves) while
  reactive's are net harmful (regret worsens) -- only regret, never the
  rejection count alone, can tell hoarding from correct anticipation apart.
- **Gate result: `passed: false` for every arm.** No non-learned pricing
  controller is activation-ready. Per the stop boundary, no learned
  predictor is reintroduced on top of any of these controllers.

## Scope and stop boundary

Complete only: this ADR; the pricing-controller interface and variants;
diagnostics; dev-set parameter selection; paired held-out evaluation with
per-scenario stratification; an honest report; full tests/coverage/typing/
lint. Stop before: reintroducing FabricPC experiments, adding any learned
prediction, changing `constraints.shadow_prices` or `SwitchCertificate`,
altering live routing, pushing, opening a PR, or updating public docs.
