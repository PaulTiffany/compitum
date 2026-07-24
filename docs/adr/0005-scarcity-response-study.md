# ADR 0005: scarcity-response phase study (tranche 4.5)

Status: accepted, observation-only. Supersedes nothing in ADR 0001-0004;
refines tranche 4's interpretation per explicit user direction.

## Reinterpretation of tranche 4

Tranche 4's frozen pacing controller cut mean regret from 2.85 to 0.36 on
held-out sequences -- but the entire effect was concentrated in 2 of 16
sequences (both the `conserve_enables_better_future` scenario), with
*exactly zero* behavioral difference from no pricing on the other 14. The
governing correction: **this is not simply another negative result.** A
pricing controller is *supposed* to stay dormant when scarcity has no
decision consequence -- byte-identical behavior to no pricing on 14 slack
-irrelevant sequences may be correct, not a defect. The open question is
narrower and more specific: do the 2 successful sequences reflect a
coherent, generalizable relationship between genuine future scarcity and
regret reduction, or merely a threshold crossed in one starkly-constructed
scenario? Tranche 4.5 is a **response-surface experiment**, not another
controller search.

## Frozen controller, varying environment

Per explicit instruction: the pacing controller configuration is frozen
at tranche 4's selected training-optimal value (`PacingController` with
`eta=1.8`, plain pacing -- tranche 4 found all four pacing-family variants
converge to statistically indistinguishable behavior once each is tuned
to its own best region, so plain pacing is used as the primary frozen
candidate, with `pacing_hysteresis`/`pacing_asymmetric`/
`pacing_bounded_smoothed` at their own tranche-4-selected frozen
parameters retained as secondary ablations). No parameter is retuned per
scarcity cell, and no cell's outcome feeds back into any parameter choice.
`ReactiveController` (tranche 3's parameters, also unchanged) and no
-pricing remain the fixed comparison points.

## Seven-axis scarcity parameterization

`src/compitum/regret_lab/scarcity_scenarios.py` introduces a single
-resource (`budget`), three-model (`conserve`, `spend`, `opportunity`)
environment, deliberately narrower than tranche 3's multi-resource
environment so the scarcity/payoff relationship can be read directly
without multi-resource-interaction confounds (which tranche 3 already
covers separately). `conserve` and `spend` are always feasible/present
(utility 1.0/2.0, consumption 1.0/`consumption_asymmetry` per step);
`opportunity` is always declared but priced/costed to be unconditionally
infeasible except during its designated window(s), where it becomes
`payoff_ratio x spend_utility` at a calibrated one-off cost -- this keeps
every case's model set identical (required by `simulate_policy`) while
still modeling a rare, high-value event.

Seven parameters, per `ScarcityParams`:

1. **`payoff_ratio`** -- `{1.0, 1.5, 3.0, 10.0}`. `1.0` is deliberately
   included as the *no real bonus* / false-scarcity control (H5): same
   budget/timing/replenishment as a real-opportunity cell, but the
   opportunity is worth no more than ordinary spending.
2. **`budget_tightness`** -- `{2.0 (slack), 1.1 (marginal), 1.0 (severe)}`,
   a multiple of the budget a perfectly conservative policy would need to
   reach and afford the opportunity exactly.
3. **`replenishment_mode`** -- `{none, partial, periodic, delayed}`: no
   replenishment; a steady partial rate; a lump sum at fixed intervals
   (known-schedule capacity, consistent with ADR 0004's assumption); or
   replenishment deferred to the second half of the sequence.
4. **`timing`** -- `{near, mid, final}`: the opportunity window at step 1,
   the midpoint, or the last step.
5. **`consumption_asymmetry`** -- `spend`'s per-step consumption relative
   to `conserve`'s fixed rate of 1.0 (default 2.0; swept at `{1.2, 2.0,
   4.0}` to test whether conserving actually frees enough capacity to
   matter).
6. **`forecast_error_mode`** -- `{none, over, under, delayed}`: the
   opportunity's `expected_consumption` understates, overstates, or is
   revealed only after `revelation_delay` steps relative to its
   `realized_consumption`.
7. **`opportunity_prevalence`** -- `{rare, moderate, stochastic}`: exactly
   the one window from `timing`; that window plus 1-2 smaller secondary
   windows (half `payoff_ratio`); or an RNG-driven independent chance of a
   small bonus opportunity at every step.

## Design: primary factorial grid + secondary one-at-a-time sweeps

A full 4x3x4x3x3x4x3 crossed factorial (payoff_ratio x budget_tightness x
replenishment x timing x asymmetry x forecast_error x prevalence) is
combinatorially unnecessary for the questions H1-H5 ask. Per the brief's
"declared factorial or space-filling design": the **primary grid** crosses
the four axes that directly define the scarcity/payoff phase space --
`payoff_ratio x budget_tightness x replenishment_mode(none, partial) x
timing` = 4x3x2x3 = 72 cells, 3 independently-seeded sequences per cell
(216 sequences) -- large enough that no conclusion rests on 1-2
sequences, directly answering tranche 4's open concern. The remaining
three axes (`consumption_asymmetry`, `forecast_error_mode`,
`opportunity_prevalence`, plus the two replenishment modes not in the
primary grid) are swept **one at a time** against a fixed reference cell
(`payoff_ratio=3.0, budget_tightness=1.1, replenishment=none,
timing=final` -- a clearly consequential-scarcity cell from the primary
grid), 5 seeds per value. Every sequence is generated from a stable,
process-independent seed (`hashlib.sha256`, per the tranche 4 fix) so the
full artifact is exactly reproducible.

## Hypotheses (preregistered, not adjusted after seeing results)

**H1, dormancy under slack:** in `budget_tightness=2.0` (slack) cells,
route disagreement rate vs. no pricing, mean regret delta, and high-value
rejections must all be approximately zero (declared tolerance: regret
delta magnitude < 0.05 per sequence on average, disagreement rate < 5%).
Pacing is **not** required to improve slack cells.

**H2, benefit under consequential scarcity:** in cells with
`payoff_ratio > 1.0` and `budget_tightness <= 1.1`, pacing must show a
paired, bootstrap-CI-significant regret reduction vs. no pricing, and this
must hold across more than one `(payoff_ratio, replenishment_mode,
timing)` combination, not only the single extreme cell tranche 4 found.

**H3, interpretable response:** engagement rate (fraction of steps with
non-negligible lambda) and regret benefit should move in a consistent
direction as `payoff_ratio` and `budget_tightness` move toward more
consequential scarcity, without requiring strict monotonicity at every
finite sample; isolated reversals or single-cell spikes are flagged and
investigated, not averaged away.

**H4, graceful boundary behavior:** near `budget_tightness ~ 1.0-1.1`
(where conservation starts to matter), regret and route-switch rate
should change smoothly across adjacent grid cells, not discontinuously.

**H5, robustness to false scarcity:** in `payoff_ratio=1.0` cells (the
opportunity never actually pays off) and in `forecast_error_mode` cells
where the anticipated cost/benefit does not materialize as expected,
pacing must not finish with large unused terminal budget after rejecting
genuinely useful routes -- it must relax back toward no-pricing behavior,
not hoard indefinitely.

## Revised gate

The frozen pacing controller passes tranche 4.5 only if: (1) it
significantly lowers regret in a preregistered consequential-scarcity
region; (2) that holds across multiple payoff/replenishment/timing
configurations, not only the original extreme case; (3) no additional hard
violations; (4) it stays within a declared small non-inferiority tolerance
in slack and false-scarcity cells; (5) the benefit is not driven by 1-2
isolated sequences (checked directly, since the primary grid uses 3 seeds
per cell across 72 cells); (6) its response near the scarcity boundary is
stable, not oscillatory; (7) its aggregate expected regret is lower under
a fixed-in-advance evaluation mixture over the primary grid's cells. If
pacing helps only above a narrow all-or-nothing threshold, it is reported
as a specialized conditional controller, not a general pricing baseline --
not reframed as a full pass.

## Reproducibility consequence

Tranche 3's originally-published numbers used the pre-fix,
process-randomized scenario hashing and remain valid as the record of
that specific internally-paired execution; they are not rewritten and are
not compared numerically against fresh reruns. All tranche 4.5 sequences
use the corrected stable-hash generator exclusively.

## Outcome (2026-07-24)

The phase-diagram pilot (`experiments/fabricpc/tranche4_5/`) ran the
pre-registered 72-cell primary grid (216 sequences) plus four secondary
sweeps, with the pacing controller frozen at tranche 4's selected
parameters throughout. Full detail in
`experiments/fabricpc/tranche4_5/REPORT.md`. Summary:

- **H2 (benefit under consequential scarcity): passed.** The improvement
  generalizes across 4 distinct `(payoff_ratio, replenishment_mode,
  timing)` configurations, not only the one extreme case tranche 4 found
  -- this directly answers tranche 4.5's governing question in the
  affirmative: the earlier result is not purely an artifact of one
  scenario.
- **H1 (dormancy under slack): failed**, but the failure is concentrated
  specifically in short-horizon (`near`/`mid` timing) slack cells and
  traced to a scenario-calibration mismatch discovered while
  investigating: `budget_tightness` is calibrated against a fully
  -conservative reference consumption rate, not against the environment's
  natural spend-preferring default behavior, which understates true slack
  at short horizons specifically. This was caught by direct inspection
  (per this project's standing practice) rather than accepted at face
  value, but was **not** corrected and re-run within this tranche --
  whether a corrected calibration would restore dormancy is an open
  question, not a settled negative about the controller itself.
- **H3 (interpretable response): passed** -- no flagged engagement
  reversals across any of the 6 sliced configurations.
- **H4 (boundary behavior): failed.** At extreme `payoff_ratio=10.0`,
  regret jumps discontinuously (by the full missed-opportunity magnitude)
  between adjacent `budget_tightness` levels in 2 of 6 configurations,
  rather than changing smoothly -- a genuine cliff, not gradual
  sensitivity.
- **H5 (robustness to false scarcity): passed** on its primary check
  (terminal unused resources match no-pricing exactly in `payoff_ratio=1.0`
  cells); a secondary diagnostic (elevated `high_value_rejections`) in
  those same cells is explained by `spend` and the non-materializing
  `opportunity` being utility-equivalent there, not by genuine waste.
- **Gate: 5 of 7 criteria passed** (significant in the consequential
  region; holds across multiple configs; no additional violations; not
  driven by isolated sequences; lower aggregate mixture regret). **Failed:
  non-inferiority in slack/false-scarcity cells (linked to the H1
  calibration issue) and stable boundary behavior (linked to H4's extreme
  -payoff cliff). `passed: false` overall.**

Per the user's own framing: the frozen pacing controller is reported as a
**specialized, conditionally-useful controller for longer-horizon,
moderate-to-high-consequence scarcity** -- not a general-purpose pricing
baseline. No parameter was retuned based on any cell's outcome; no learned
predictor was reintroduced.

## Tranche 4.6 addendum: calibration rerun and cliff diagnostic (2026-07-24)

Per user direction, before any FabricPC reintroduction, both open items
from the tranche 4.5 outcome were resolved directly (no controller
tuning). Full detail in `experiments/fabricpc/tranche4_6/REPORT.md`.

**Corrected-slack rerun:** recalibrating `budget_tightness` against the
natural spend-preferring rate (rather than the fully-conservative one)
roughly halves route disagreement and regret impact across the near/mid
-timing slice, but does not eliminate it -- H1's failure was **partially**
a scenario-calibration artifact and **partially** a genuine, unresolved
property of the frozen pacing controller's short-horizon behavior. Both
the original and corrected datasets/results are retained; tranche 4.5's
report is not rewritten.

**Cliff diagnostic:** densely sampling absolute budget at the finest
available grid resolution around the `payoff_ratio=10.0` boundary found
the H4 cliff is not one mechanism. Of the 2 originally-flagged
configurations, 1 (`none, mid`) is a genuine, pacing-created capability
(no-pricing never captures the opportunity anywhere sampled; pacing
transitions cleanly at one grid step -- the sharpness is intrinsic to a
one-shot binary event, not evidence of instability or under-sampling), and
1 (`partial, near`) is a shared, policy-independent feasibility threshold
unrelated to pacing's price dynamics. A third, previously-unflagged
configuration (`partial, mid`) revealed a genuine narrow non-monotonicity:
pacing captures the opportunity at one budget level, loses it at the very
next grid step, then recaptures it at every level beyond -- a real,
reproducible controller fragility tranche 4.5's coarser two-point
comparison could not detect.

**Also found and reported (not fixed):** `opportunity_prevalence="rare"`
(every primary-grid cell) never consumes its RNG argument, so tranche
4.5's "3 seeds per cell" are byte-identical duplicates for the primary
grid, not independent draws. Does not invalidate H2's cross-configuration
finding, but affects how "not driven by isolated sequences" should be read
for primary-grid analyses.

**Consequence for tranche 5:** the residual corrector's base controller is
not a clean reference -- it carries a known, only-partially-understood
non-dormancy in short-horizon slack cells and at least one known narrow
non-monotonic region. Tranche 5's non-inferiority and boundary-stability
gate criteria inherit these imperfections as their baseline.

## Scope and stop boundary

Complete only: this ADR; the scarcity scenario generator; the frozen
-controller phase-diagram pilot; the preregistered hypothesis and gate
evaluation; a scenario/region-stratified report with phase-map tables;
full tests/coverage/typing/lint. Stop before: retuning any parameter based
on a held-out cell's outcome; reintroducing FabricPC or any learned
predictor; production pricing integration; changes to
`constraints.shadow_prices` or `SwitchCertificate`; push or PR.
