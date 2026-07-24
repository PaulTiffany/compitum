# Tranche 4.5: frozen-controller scarcity phase-diagram study — report

Branch `experiment/fabricpc-trajectory-observer`, cut from tag `v0.2.0`.
Pure `compitum.regret_lab` -- no FabricPC, no JAX, **no parameter tuning
anywhere in this tranche**: the pacing controller (and its three sibling
variants) run at exactly the parameters tranche 4 selected from its own
development data. Nothing here affects route selection, `main`, the
`v0.2.0` tag, `SwitchCertificate`, or `constraints.shadow_prices`.

## The question this tranche answers

Tranche 4's frozen pacing controller cut mean regret 8x on held-out
sequences, but the entire effect was 2 of 16 sequences, both the
`conserve_enables_better_future` scenario. Per the user's reframing: exact
non-engagement on the other 14 may be *correct* dormancy, not a defect --
the open question was whether the 2-sequence win reflects a coherent,
generalizable scarcity/opportunity-cost relationship or a threshold
crossed in one extreme construction. This is a response-surface
experiment: the controller is frozen, the environment varies across a
declared 72-cell primary grid (`payoff_ratio x budget_tightness x
replenishment_mode x timing`, 3 seeds/cell = 216 sequences) plus four
one-at-a-time secondary sweeps (65 more sequences), per
`docs/adr/0005-scarcity-response-study.md`.

## Headline: a genuine, generalizable benefit exists -- but this is not yet a general-purpose controller

**H2 passed: the benefit generalizes.** Restricting to consequential
-scarcity cells (`payoff_ratio > 1.0` and `budget_tightness <= 1.1`), the
frozen pacing controller shows a meaningful regret improvement
(`delta < -0.1`) in **4 distinct `(payoff_ratio, replenishment_mode,
timing)` configurations**, not only the single extreme case tranche 4
found: `(1.5, partial, near)`, `(3.0, partial, mid)`, `(10.0, none, mid)`,
`(10.0, partial, mid)`. This directly answers tranche 4.5's governing
question: **the tranche 4 result is not purely an artifact of one
scenario.**

**Phase map (mean paired regret delta vs. no pricing, averaged over
replenishment mode and timing; negative = pacing helps):**

| payoff_ratio \ budget_tightness | 2.0 (slack) | 1.1 (marginal) | 1.0 (severe) |
| --- | --- | --- | --- |
| 1.0 (no real bonus) | 0.0 | 0.0 | 0.0 |
| 1.5 | -1.0 | -0.33 | 0.0 |
| 3.0 | **+0.5** | +0.17 | -0.17 |
| 10.0 | +0.33 | **-5.0** | -2.5 |

Two things stand out. First, at `payoff_ratio=10.0` the biggest benefit is
at **marginal**, not severe, tightness (-5.0 vs -2.5): severe cells sit
exactly at the fully-conservative minimum, where even pacing's
conservatism has no margin to reliably close the gap (this connects
directly to the H4 finding below). Second, `payoff_ratio=3.0` at slack
tightness shows pacing **hurting** (+0.5) -- the first hint of the H1
finding.

## H1 (dormancy under slack): failed, but the failure has a specific, traced cause

Averaged across all 24 slack cells, every arm shows real engagement
(pacing: mean |regret delta| 0.46, mean route-disagreement rate 29%) --
failing the preregistered dormancy tolerance (0.05 for both). Rather than
report this as "the controller doesn't rest," direct inspection (per this
project's standing practice of checking a suspicious result before
trusting it) traced the failure to a **specific, narrow cause**: every
slack cell showing disagreement has `timing=near` or `timing=mid` --
**not one** `timing=final` slack cell shows any disagreement at all (spot
-checked directly: a `payoff_ratio=3.0, budget_tightness=2.0,
timing=final` sequence shows lambda pinned at exactly `0.0` for all 12
steps, byte-identical choices to no pricing).

The mechanism: `budget_tightness` is calibrated against a fully
-conservative reference rate (`t_opp x CONSERVE_RATE + OPPORTUNITY_COST`),
but the environment's natural default behavior (and no-pricing's actual
behavior) prefers `spend` at twice that rate. At short horizons
(`near`: `t_opp=1`, `mid`: `t_opp=6`), the pacing target rate
(`total_available / total_steps`) is diluted enough by the small number of
pre-opportunity steps that a `spend`-preferring policy looks like
over-consumption to the pacing controller even at nominal `budget_tightness=2.0`
-- a scenario-calibration mismatch, not necessarily a controller defect.
**This was not corrected and re-run within this tranche** -- whether a
`spend`-rate-calibrated `budget_tightness` would restore dormancy at short
horizons is an open question for the next tranche, not a settled claim
either way.

## H3 (interpretable response): passed

Across all 6 `(replenishment_mode, timing)` slices, sorting by decreasing
budget tightness then increasing payoff ratio, engagement rate never drops
by more than the noise threshold (0.2) as scarcity/payoff increases --
**zero flagged reversals**. Whatever else is true, the controller does not
respond backwards to more consequential scarcity.

## H4 (boundary behavior): failed -- a real cliff at extreme payoff

Comparing `budget_tightness=1.1` to `1.0` within the same
`(payoff_ratio, replenishment_mode, timing)` slice: at `payoff_ratio=10.0`,
regret jumps by the **full missed-opportunity magnitude (15.0)** between
adjacent tightness levels in 2 of 6 configurations (e.g.
`(10.0, none, mid)`: regret 0.0 at marginal, 15.0 at severe). This is not
gradual sensitivity -- it is an all-or-nothing threshold effect,
concentrated specifically at the most extreme payoff ratio tested. At
`payoff_ratio<=3.0`, no comparable jump occurs (max magnitude there is
1.0). **Max jump magnitude across all slices: 15.0**, well above the
preregistered stability threshold (5.0).

## H5 (robustness to false scarcity): passed on its primary check

In the 18 `payoff_ratio=1.0` cells (the opportunity never actually pays
more than ordinary spending), mean terminal unused budget for pacing
matches no-pricing **exactly** (delta = 0.0) -- pacing does not finish
these sequences hoarding unused resources after rejecting genuinely useful
routes. A secondary diagnostic, `high_value_rejections`, is elevated
(+13.3 on average) in these same cells; this is not a contradiction --
under `payoff_ratio=1.0`, `spend` and the (never-better) `opportunity` are
utility-equivalent, so "rejecting" one for the other costs nothing in
practice, and the matching terminal-resource numbers confirm no real harm
resulted.

## Secondary sweeps (reference cell: `payoff_ratio=3.0, budget_tightness=1.1, replenishment=none, timing=final`)

| axis | mean paired delta vs. no pricing | note |
| --- | --- | --- |
| consumption_asymmetry ∈ {1.2, 2.0, 4.0} | -1.33 | pacing helps more as spend/conserve cost gap widens |
| forecast_error_mode ∈ {none, over, under, delayed} | 0.0 | no measurable sensitivity to forecast accuracy at this cell |
| opportunity_prevalence ∈ {rare, moderate, stochastic} | +0.33 | mild degradation with multiple/stochastic opportunities |
| replenishment_mode ∈ {periodic, delayed} | 0.0 | no measurable difference from the primary grid's none/partial |

## Revised gate: 5 of 7 criteria passed

| # | criterion | result |
| --- | --- | --- |
| 1 | significant in consequential region (bootstrap CI) | **passed** (CI `[-2.15, -0.58]`, entirely negative) |
| 2 | holds across multiple configs | **passed** (4 distinct configs) |
| 3 | no additional violations | **passed** |
| 4 | non-inferior in slack/false-scarcity | **failed** (linked to H1's calibration issue) |
| 5 | not driven by isolated sequences | **passed** (5 improving cells x 3 seeds = 15 sequences) |
| 6 | stable near boundary | **failed** (linked to H4's extreme-payoff cliff) |
| 7 | lower aggregate mixture regret (uniform weight over all 72 cells) | **passed** |

**`activation_gate.passed: false`.** Per the user's own framing for this
outcome shape: the frozen pacing controller is reported as a
**specialized, conditionally-useful controller for longer-horizon,
moderate-to-high-consequence scarcity** -- not a general-purpose pricing
baseline ready for broader activation.

**Latency:** the entire study (216 primary + 65 secondary sequences,
6 arms, hindsight computation, phase-map/hypothesis/gate evaluation) runs
in well under one second, pure Python/numpy.

**Baseline integrity:** full worktree suite unchanged from tranche 4.5.2's
commit (593 passed, 1 skipped, 1 pre-existing unrelated worktree failure);
`src/compitum` remains 100.00% line+branch covered, mypy `--strict` clean.

## Honest methodology notes

- No parameter was retuned based on any phase-diagram cell's outcome.
  `ReactiveController` and all four `PacingController` variants use
  exactly tranche 4's selected values, read directly from that tranche's
  `pilot_report.json`.
- The H1 calibration mismatch is a property of the *scenario generator's*
  `budget_tightness` definition, discovered by direct inspection, not
  assumed. It was traced precisely (short-horizon cells only; zero
  `timing=final` slack cells affected) but not fixed in this tranche.
- The H4 cliff and the phase map's `payoff_ratio=10.0` marginal-beats
  -severe pattern are consistent with the same underlying mechanism: at
  the exact conservative minimum, there is no margin left for *any*
  policy, pricing included, to reliably close the gap.
- Single-resource, three-model environment (`conserve`/`spend`/
  `opportunity`), deliberately narrower than tranche 3's multi-resource
  environment, isolates the scarcity/payoff relationship but does not
  test multi-resource interaction with scarcity phase effects together.

## Since the gate did not pass

No learned predictor is reintroduced. `constraints.shadow_prices`,
`SwitchCertificate`, routing behavior, and the `v0.2.0` tag remain
untouched.

## Open items and smallest defensible next step

Two concrete, well-understood open items, both scenario-generator issues
rather than controller behavior in question:

1. Recalibrate `budget_tightness` against the natural (`spend`-preferring)
   reference rate rather than the fully-conservative one, and re-run the
   `near`/`mid`-timing slack cells specifically to determine whether H1's
   failure was a scenario artifact or a real property of the controller.
2. Investigate the `payoff_ratio=10.0` cliff directly: is there a
   continuous, narrower band of `budget_tightness` values between 1.0 and
   1.1 where the transition is gradual, or is this genuinely a step
   function at this payoff scale?

Only once these are resolved does it make sense to ask whether a simple
non-FabricPC predictor, and then FabricPC, can improve this controller
further -- and even then, strictly in the longer-horizon,
moderate-to-high-consequence region this study found it to actually help.
