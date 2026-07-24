# Tranche 4: pricing-controller repair — report

Branch `experiment/fabricpc-trajectory-observer`, cut from tag `v0.2.0`
(`a8de8cbafa5eb00b523f539c340ba81a146aa781`). Pure `compitum.regret_lab` --
no FabricPC, no JAX involved anywhere in this tranche, per the standing
instruction to keep FabricPC frozen until a non-learned pricing baseline
beats no pricing. Nothing in this tranche affects route selection, `main`,
the `v0.2.0` tag, the frozen `SwitchCertificate` schema, or
`constraints.shadow_prices`.

## The finding this tranche was built to address

Tranche 3: mean regret static/no-pricing 2.778, reactive dual controller
2.982, dual+EWMA/FabricPC ~3.67. **Bad pricing is worse than no pricing.**
Per user direction, this tranche is exclusively about establishing a
non-learned pricing controller that actually beats no pricing -- before
any learned predictor is reintroduced -- and distinguishing genuine
resource preservation from good allocation.

## What was built

- **`src/compitum/regret_lab/pricing.py`** -- `PricingUpdateContext` (raw
  per-step ingredients), a `PricingController` protocol, `ReactiveController`
  (adapter reproducing tranche 3's exact reactive formula around the
  byte-for-byte-unchanged `DualController`), and one flexible
  `PacingController` class realizing four named variants via parameters:
  plain pacing, pacing+hysteresis/deadband, asymmetric relaxation, and
  EMA-smoothed/bounded-step. `simulate_policy`'s hook was generalized from
  `dual_controller` to `pricing_controller` against this interface.
- **`src/compitum/regret_lab/diagnostics.py`** -- `conservation_depletion_split`,
  an explicitly-labeled heuristic attributing per-step regret (vs. the
  hindsight oracle's own per-step choice) to conservation (resources were
  not genuinely scarce) or depletion (they were), with the residual
  reported honestly rather than hidden.
- `PolicyRunResult` gained `terminal_remaining` and `high_value_rejections`
  (a direct, per-step signature of hoarding: a genuinely-affordable,
  higher-utility model was available but pricing picked something worse).
- **`experiments/fabricpc/tranche4/run_pricing_pilot.py`** -- dev-set grid
  search on training sequences (frozen before touching test sequences),
  then a six-arm paired held-out evaluation with bootstrap CIs,
  scenario-stratified regret, conservation/depletion diagnostics, and
  controller price volatility.
- 19 new tests, 100% line+branch coverage maintained across `regret_lab`,
  mypy `--strict` clean.

## A real bug found and fixed along the way

`generate_dynamic_dataset` seeded its per-scenario RNG stream with
`hash(scenario) & 0xFFFFFFFF` -- Python's built-in string `hash()`, which
is randomized per process (PEP 456) unless `PYTHONHASHSEED` is fixed. This
silently broke the advertised "deterministic, reproducible dataset
generator given the same seed" guarantee **across separate process runs**
(it was, and remains, correct *within* one process, which is why every
prior test suite run and every prior tranche's single pilot execution was
internally self-consistent). Caught by direct empirical comparison: this
tranche's fresh `no_pricing`/`reactive` arm regret (2.851/3.193) differed
slightly from tranche 3's stored `static`/`dual_no_predictor` numbers
(2.778/2.982) despite using the identical `seed=2026` -- a gap too
structured to be noise, confirmed by printing `hash("permanently_slack")`
twice in separate processes and observing different values. Fixed with a
`hashlib.sha256`-based stable hash (matching the pattern already used
elsewhere in this codebase for exactly this reason); a regression test
hardcodes the expected stable values so a reversion would be caught
immediately. Tranche 2 and 3's published numbers remain valid as the
record of those specific runs; they are simply not exactly reproducible by
re-running those scripts fresh, which is now fixed going forward.

## Pilot results (8 scenarios × 4 sequences × 8 steps; 16 train / 16 test sequences)

**Reactive controller (tranche 3's reference, parameters unchanged):**
paired regret delta vs no pricing **+0.341**, 95% CI `[0.065, 0.648]` --
entirely positive. Reproduces "bad pricing is worse than no pricing" under
the corrected, now-genuinely-reproducible dataset.

**A too-narrow first grid, caught before trusting it.** The first
parameter search (eta up to 1.0 for plain pacing) found every pacing
-family variant statistically indistinguishable from no pricing --
suspiciously flat. Rather than accept that, a direct training-sequence
sweep to wider eta values found mean regret falling from ~2.86 to ~0.34 at
eta ≈ 1.2-2.1, a region the original grid never touched. All four grids
were widened and the pilot re-run; the bounded/smoothed family needed a
correspondingly larger `max_step` (up to 2.0, not 0.1-0.5) to reach the
same region. This is the second time in this program a first-pass grid or
threshold turned out to be wrong before being caught by direct
verification (tranche 2's classification threshold was the first) --
worth naming as a recurring risk in this kind of work.

**With properly widened grids, all four pacing-family arms achieve mean
regret 0.35-0.36 vs no pricing's 2.85** (paired delta -2.50, 95% CI
`[-6.26, 0.0]`). This looks like a dramatic win on the mean alone. It is
not a broad one:

| scenario | no pricing | pacing (representative) | reactive |
| --- | --- | --- | --- |
| conserve_enables_better_future | 20.63 | **0.66** | 20.63 |
| single_resource_scarce_period | 1.68 | 1.68 | 1.68 |
| delayed_realization | 0.50 | 0.50 | 1.73 |
| multi_resource_interaction | 0.0 | 0.0 | 1.50 |
| demand_burst, permanently_slack, forecast_error, premature_conservation_regret | ~0 | ~0 (identical) | ~0 (identical) |

**The entire improvement is concentrated in exactly 2 of 16 test
sequences** -- both `conserve_enables_better_future`, the one scenario
whose design (a single enormous, un-repeatable payoff at the final step,
funded by a budget that never replenishes) is exactly what pacing-style
conservation is built to protect. On the other 14 sequences across the
remaining 7 scenarios, every pacing-family arm is **byte-for-byte
identical** to no pricing -- not merely similar, exactly equal, meaning
lambda never rises enough (or the utility gaps are too large relative to
achievable price penalties) to change a single decision. This is why the
paired bootstrap CI's upper bound sits exactly at `0.0` rather than
comfortably below it: with 14 of 16 paired deltas exactly zero, a
substantial fraction of bootstrap resamples never draw either of the two
informative sequences, pushing the 97.5th percentile to the boundary. The
gate is designed to guard against exactly this shape of result (item 4 in
the pre-registration: "does not merely shift regret into a small number of
sequences") and it does so correctly here, in reverse -- the improvement,
not a regression, is what's concentrated, and the gate rightly declines to
call a two-sequence effect a general one.

**Resource preservation vs. good allocation, concretely.** `pacing` and
`reactive` both show `high_value_rejections = 8` (both reject a genuinely
-affordable, higher-base-utility model 8 times across the test set). For
`pacing` this is net beneficial (mean regret drops from 2.85 to 0.36); for
`reactive` it is net harmful (mean regret rises from 2.85 to 3.19). The
rejection count alone cannot distinguish correct anticipatory conservation
from pure hoarding -- only regret can, which is exactly why this tranche's
ADR insisted regret remain the primary metric and terminal-resource/
rejection counts stay diagnostic, never a proxy target.

**Activation gate: `passed: false` for every arm.** No non-learned pricing
controller is activation-ready by the pre-registered, paired,
bootstrap-CI-significant standard.

**Latency:** pure Python/numpy, no external calls; the entire pilot
(dataset generation, 47 grid configs across four families on 16 training
sequences, hindsight computation, six-arm evaluation, diagnostics) runs in
well under one second.

**Baseline integrity:** full worktree suite: 569 passed, 1 skipped (known
Windows subprocess skip), 1 pre-existing failure unrelated to this tranche
(the worktree-`.git`-is-a-file `git_commit_short` issue, unchanged from
tranches 2-3). `src/compitum` (including `regret_lab`) remains 100.00%
line+branch covered, mypy `--strict` clean, ruff clean.

## Honest methodology notes

- Pacing-family parameters were selected on training sequences only,
  scored by mean regret subject to zero increase in total violations
  relative to the no-pricing reference, then frozen before any held-out
  evaluation. The reactive controller's parameters were **not** tuned --
  it is retained exactly as tranche 3 left it, a fixed comparison point,
  not a candidate.
- The grids searched are declared in full in
  `experiments/fabricpc/tranche4/artifacts/pilot_report.json`'s
  `parameter_selection` section, including the final widened ranges.
- Six of eight controlled scenarios show **zero** measurable engagement
  from any pricing controller tested (identical choices to no pricing).
  Whether this reflects genuinely generous resource margins in those
  scenario designs, or a limitation of the cumulative-usage-vs-horizon
  -target pacing error formulation specifically, is not resolved by this
  pilot.
- `total_available_over_horizon` treats the full replenishment schedule as
  known capacity (see ADR 0004); this assumption is unchanged from
  design and was not revisited here.

## Since the gate did not pass

No learned predictor (EWMA, FabricPC, or anything else) is reintroduced.
`constraints.shadow_prices`, `SwitchCertificate`, routing behavior, and the
`v0.2.0` tag remain untouched.

## Open items and smallest defensible next step

Unresolved: whether the near-total absence of pricing engagement in 6 of 8
scenarios reflects the environment's resource margins being too generous
by design, or a real limitation of this specific pacing-error formulation
that a differently-shaped error signal (e.g. reacting to short-horizon
projected shortfall rather than a full-horizon average rate) might not
share. The one scenario where pacing helps enormously is also the one
built to have the starkest, most extreme structure (a single payoff 5-10x
larger than any other utility in the sequence, funded by a
never-replenishing resource) -- it is not yet known whether pacing would
help in a *milder*, more realistic version of the same dynamic (a payoff
2x rather than 10x larger, or partial replenishment).

Smallest defensible next step, still not reintroducing any learned
component: before concluding pacing "doesn't generalize," test it against
scenario variants that vary the STRENGTH of the conserve-now-enables
-better-future dynamic (several magnitudes of final payoff, several
replenishment rates) to map out whether the 2-sequence win reflects a
narrow all-or-nothing threshold effect or a real, continuous relationship
this pilot's specific scenario design happened to sample only at its
extreme. Only once a non-learned controller shows a *broadly* distributed,
gate-passing improvement does it make sense to ask whether a simple
non-FabricPC predictor, and then FabricPC, can improve it further.
