# Tranche 5: FabricPC residual shadow pricing — report

Branch `experiment/fabricpc-trajectory-observer`, cut from tag `v0.2.0`.
FabricPC pin unchanged: v0.3.2 at `32ae295182ab944b8f084abaf4a40da2c50bab5f`
(external checkout `C:\src\FabricPC`; not vendored, not patched). Nothing
in this tranche affects route selection, `main`, the `v0.2.0` tag,
`SwitchCertificate`, or `constraints.shadow_prices`.

## The question this tranche answers

Per the governing rule: "the deterministic pacing controller supplies the
price; FabricPC may earn the right to supply a bounded, prospective
correction -- but only if that correction reduces constrained regret
beyond both pacing and an ordinary sequential predictor." Tranche 4/4.6
established the frozen pacing controller (not the reactive controller) as
the real baseline, with known, carried-forward imperfections (partial
short-horizon non-dormancy, one narrow non-monotonic region). This tranche
reintroduces FabricPC strictly as a bounded, gated corrector of that
baseline's price and asks whether the correction earns its place.

## What was built

- **`src/compitum/regret_lab/residual_target.py`** -- `LambdaInterval` and
  `oracle_price_residual`: an exact, closed-form (not estimated) computation
  of the set of prices that would reproduce the hindsight oracle's own
  per-step choice, and the minimal signed nudge into that interval. Used
  only to build offline training targets; infeasible rows (e.g. the oracle
  deferred) are excluded, never treated as zero.
- **`src/compitum/regret_lab/residual_channels.py`** -- a declared 13-dim
  per-step channel (remaining resources, pacing error, replenishment,
  per-model expected consumption, utility gap, current/changed price,
  route-switch indicator, horizon fraction, opportunity-recency proxy,
  forecast error), with history bookkeeping.
- **`src/compitum/regret_lab/residual_pricing.py`** -- `ResidualPricingController`:
  wraps the frozen `PacingController` unchanged, maintains a bounded
  5-step window of channel vectors, clips any predicted correction to
  `+-2.0`, gates it (`lambda_base > 0.01`, a simple declared
  regional-scope proxy), and degrades deterministically to zero on any
  predictor exception -- every step produces a `ResidualCorrectionRecord`
  (status, magnitude, window snapshot) for provenance.
- **`src/compitum/regret_lab/windowed_predictor.py`** -- generic ridge
  regression + window-flattening, used both directly (arm 3) and as the
  offline fit mapping FabricPC trajectory features to a residual (arms 4-7).
- **`experiments/fabricpc/tranche5/fabricpc_residual_observer.py`** --
  JAX-side observer over a genuine multi-step window (flattened last 5
  steps, source dim 65) rather than one static snapshot --
  source(65)->hidden(16)->latent(6), reusing tranches 1-3's pinned-receipt
  pattern and raw schema unchanged.
- **`experiments/fabricpc/tranche5/run_residual_shadow_pilot.py`** --
  offline training-row collection via a permanently-zero "recorder"
  controller (so recorded decisions exactly match plain frozen pacing,
  never influencing anything), oracle-compatible target computation, three
  frozen ridge fits, and the seven-arm paired held-out evaluation.
- A real bug found and fixed while building this: `simulate_policy` only
  calls `pricing_controller.update()` on non-deferred steps, so a
  controller's `records` list is shorter than the sequence and **must be
  indexed by each record's own `.step` field, never by list position** --
  caught immediately via a small-scale smoke test before the full run,
  not discovered only after a confusing result.
- 46 new tests across the new modules, 100% line+branch coverage
  maintained, mypy `--strict` clean.

## Pilot results (72-cell primary grid, 48 train / 24 test cells, 1 sequence per cell)

**FabricPC's correction is genuinely inert, not harmful.**

| arm | mean regret | route disagreement vs. frozen pacing | mean \|correction\| | clip rate |
| --- | --- | --- | --- | --- |
| no pricing | 2.250 | -- | -- | -- |
| frozen pacing (baseline) | **2.0833** | -- | -- | -- |
| + non-FabricPC windowed ridge | 2.125 | 5.2% | 1.449 | **58.5%** |
| + FabricPC terminal-state residual | 2.0833 | 1.4% | 0.353 | 0% |
| + FabricPC trajectory residual | 2.0833 | 1.4% | 0.345 | 0% |
| + FabricPC trajectory, shuffled control | 2.125 | 5.2% | 0.409 | 0.5% |
| + FabricPC trajectory, gate forced open | 2.0833 | 1.4% | 0.336 | 0% |

The three FabricPC arms (terminal, trajectory, gate-forced-open) all
produce **exactly the same mean regret as frozen pacing alone, to the
decimal** -- not merely similar, identical. They compute and apply real,
nonzero corrections every gated step (0% clipped, 0% failed -- the
predictor never refuses and is never out of bounds), but those corrections
essentially never cross a decision-relevant threshold: only 1.4% of steps
differ from what frozen pacing alone would have chosen. **The correction
exists; it just doesn't do anything.**

The two arms whose corrections *do* change decisions more often (5.2%
disagreement, roughly 4x the FabricPC arms') make things **worse**, not
better: the non-FabricPC windowed ridge and the shuffled-trajectory
control both land at 2.125 -- worse than frozen pacing. The windowed
ridge's raw predictions are clipped to the `+-2.0` bound **58.5% of the
time**, meaning its unclipped output is frequently far outside a sane
range; the magnitude cap is doing real, necessary work there, not sitting
idle as a formality.

**Activation gate: fails cleanly.**

| criterion | result |
| --- | --- |
| beats frozen pacing | **false** -- paired delta exactly 0.0 |
| beats non-FabricPC windowed predictor | false (delta -0.042, CI `[-0.125, 0.0]` -- does not exclude zero, and arm 3 itself underperforms the baseline, so this is a low bar even when nominally cleared) |
| beats shuffled control | false (identical numbers to the row above) |
| no additional violations | **true** -- the only criterion satisfied |

`activation_gate.passed: false`. No claim of improvement is made. FabricPC
has not earned the right to supply a production-relevant correction under
this pilot's design.

**Conservation/depletion diagnostic** confirms the FabricPC arms track
frozen pacing almost exactly (conservation-regret 4.79 vs. pacing's 4.875;
depletion-regret 3.71 vs. 3.625) -- consistent with the correction being
inert rather than reshaping behavior in either direction. The windowed
-ridge and shuffled arms shift slightly toward *more* conservation-regret
without a matching depletion-regret improvement -- their larger,
occasionally-clipped corrections cost a little extra caution without
buying anything back.

**Latency:** 1640 real FabricPC observations (410 training x 2 feature
extractions + 1230 test-time calls across 4 FabricPC arms), p50 0.49s, p95
0.66s, max 2.24s, 0 governed failures. Total pilot runtime ~723s.

**Baseline integrity:** `src/compitum` remains 100.00% line+branch
covered, mypy `--strict` clean, ruff clean (verified before this run).

## Honest methodology notes

- Training rows: 410 of 576 possible (48 cells x 12 steps) had a feasible
  oracle-compatible interval; the other 166 (mostly deferred-oracle or
  strictly-dominated-choice steps) were excluded, never treated as a zero
  target.
- Both non-FabricPC-windowed (arm 3) and FabricPC (arms 4/5/7) predictors
  are simple ridge regressions with no hyperparameter search, matching
  every prior tranche's methodology -- the windowed ridge's heavy clipping
  rate suggests this simple approach may be poorly suited to the raw,
  unprocessed window features specifically (65 raw, largely-redundant
  dimensions for a ridge fit on ~400 rows), a different failure mode than
  FabricPC's inertness.
- The regional-scope gate (`lambda_base > 0.01`) is a simple, declared
  proxy for "pacing itself perceives some scarcity here" -- not derived
  from scenario metadata directly. Arm 7 (gate forced open everywhere)
  shows **identical** results to arm 5 (gated), meaning the gate was not
  the limiting factor here -- the correction is inert whether or not it is
  allowed to apply outside the nominally consequential region.
- This is the second time in this program that a residual/correction
  mechanism has landed at one of two extremes -- near-zero/inert (safe but
  useless) or large and occasionally miscalibrated (harmful when not
  strictly bounded) -- rather than a well-calibrated, genuinely useful
  middle. Worth naming as a pattern for future work on this class of
  correction, not just a one-off result.

## Since the gate did not pass

No learned predictor is activated. `constraints.shadow_prices`,
`SwitchCertificate`, routing behavior, and the `v0.2.0` tag remain
untouched. The frozen pacing controller (with its known tranche-4.6
imperfections) remains the only pricing mechanism with any demonstrated
benefit in this program.

## Open items and smallest defensible next step

Unresolved: whether FabricPC's inertness reflects (a) genuinely
insufficient signal in the declared 13-dim x 5-step window for this
correction task, (b) the ridge-regression fit being too simple to extract
what signal exists (unlike the windowed predictor, FabricPC's corrections
were never clipped, suggesting its OWN outputs stayed conservative rather
than being reined in -- worth checking whether the ridge fit on FabricPC's
trajectory features is itself under-fitting), or (c) 410 training rows
being too few for either predictor to learn a reliable correction at this
granularity.

Smallest defensible next step, still not claiming any improvement: before
concluding FabricPC "has no signal" for this task, check whether the
ridge-fit trajectory-feature model's own held-out prediction error (not
regret -- a direct, non-decision-mediated check) shows ANY better-than
-constant accuracy on the 410 training-derived oracle residual targets,
using proper cross-validation rather than the single train/test split
used here. If even the direct numerical fit is no better than predicting
the mean residual, that would indicate the window genuinely lacks the
needed signal (option a/c above); if the direct fit shows real skill but
regret is still unchanged, that points to option (b) or to a decision
-boundary insensitivity specific to this environment's granularity, not to
FabricPC's features being uninformative.
