# Tranche 4.6: corrected-slack rerun + extreme-payoff cliff diagnostic — report

Branch `experiment/fabricpc-trajectory-observer`. Pure `compitum.regret_lab`
-- no FabricPC, no JAX, **no controller tuning**. Both open questions from
tranche 4.5 are resolved here before any FabricPC work, per the standing
instruction. Original tranche 4.5 artifacts (`experiments/fabricpc/tranche4_5/`)
are untouched; this tranche's artifacts are separate and additive.

## Part 1: corrected-slack rerun

**Question:** was tranche 4.5's H1 failure (29% route disagreement, 0.46
mean |regret delta| in near/mid-timing slack cells) a scenario-calibration
artifact, or genuine controller over-engagement?

**Answer: both, partially.** Recalibrating `budget_tightness` against the
natural spend-preferring rate (`consumption_asymmetry`, default 2.0)
instead of the fully-conservative rate (1.0) reduces, but does not
eliminate, the effect:

| | slack cells only | all near/mid cells |
| --- | --- | --- |
| original: mean \|regret delta\| | 0.6875 | 1.25 |
| corrected: mean \|regret delta\| | 0.625 | 0.646 |
| original: mean route disagreement | 43.2% | 46.5% |
| corrected: mean route disagreement | 34.4% | 36.3% |

The correction roughly **halves** the disagreement/regret-impact across
the full near/mid slice, and reduces the slack-only cells more modestly
(~9-20%). **This means the calibration mismatch was real and accounts for
a meaningful share of H1's failure, but a substantial residual
non-dormancy remains even under the corrected reference** -- roughly a
third of steps in nominally-slack, short-horizon cells still show route
disagreement from no pricing. H1 should be read as: partially a scenario
-generator artifact (now understood and quantified), partially a genuine
property of the frozen pacing controller's short-horizon behavior that
recalibration alone does not resolve.

## Part 2: extreme-payoff cliff diagnostic

**Question:** for `payoff_ratio=10.0`, is the H4 regret cliff between
`budget_tightness=1.1` and `1.0` caused by discrete feasibility, discrete
route selection, price dynamics, or inadequate sampling resolution?

Sampled absolute initial budget at the finest available grid resolution
(`GRID_UNIT=0.25` -- confirmed this is the true resolution floor: sampling
the `budget_tightness` *ratio* densely aliases onto the same quantized
budget values, since the DP requires exact 0.25 multiples; sampling
absolute budget directly is the only way to get genuinely distinct
sequences) across both flagged configurations and 4 unflagged ones for
contrast. **The answer is not one mechanism -- it is a mix, and which one
applies differs by configuration:**

**Case 1 -- a genuine, pacing-created capability (`replenishment=none,
timing=mid`, flagged):** no-pricing **never** captures the opportunity
anywhere in the sampled range `[10.5, 12.75]` (regret pinned at 15.0
throughout). Pacing transitions cleanly from missing (regret 15.0, budget
`< 12.0`) to fully capturing (regret 0.0, budget `>= 12.0`) at exactly one
grid step. This is real: pacing creates a capability no-pricing does not
have anywhere in this range. The sharpness is intrinsic to the
environment, not an instability -- capturing a one-shot, all-or-nothing
opportunity is a binary event, so *any* policy's regret-vs-budget curve is
a step function here, at whatever resolution you sample it. This is not
"inadequate sampling resolution": 0.25 is the finest resolution the
environment supports, and the transition genuinely occupies exactly one
step at that resolution.

**Case 2 -- a shared, policy-independent feasibility threshold
(`replenishment=partial, timing=near`, flagged; also
`replenishment=none, timing=near`, unflagged):** no-pricing and pacing
transition at **the identical budget value** in both configs. This "cliff"
is not about pacing's price dynamics at all -- it is the same discrete
feasibility boundary every policy hits, confirming these two flagged cases
are not really about controller behavior.

**Case 3 -- a genuine, narrow non-monotonicity (`replenishment=partial,
timing=mid`, unflagged, found while sampling for contrast):** pacing
captures the opportunity at budget `10.5` (regret 0.0), **fails** at
`10.75` (regret 15.0, an isolated single-grid-point dip), then captures
again at every point from `11.0` upward. More budget causing pacing to
*lose* the opportunity it held with slightly less budget is a real,
reproducible anomaly in the controller's dynamics (the pacing target rate
itself scales with `total_available`, so a small budget change shifts the
whole lambda trajectory, not just the feasibility margin) -- distinct from
cases 1 and 2, and the closest thing to genuine instability found in this
diagnostic.

**Summary: of the 2 originally-flagged configurations, 1 (`none, mid`) is
a genuine and desirable pacing effect, and 1 (`partial, near`) is a shared
environmental threshold unrelated to pacing specifically.** The
diagnostic also surfaced 1 additional, previously-unflagged genuine
controller anomaly (`partial, mid`) that tranche 4.5's coarser
`{1.0, 1.1}` two-point comparison could not have detected.

## A methodology finding, verified before use

While building this diagnostic: `opportunity_prevalence="rare"` (every
primary-grid cell in tranches 4.5 and 4.6) never consumes its `rng`
argument inside `build_scarcity_sequence` -- confirmed directly (two
different seeds produce byte-identical sequences for a rare-prevalence
cell). **Tranche 4.5's "3 seeds per cell" are byte-identical duplicates for
the primary grid, not independent draws.** This does not invalidate
tranche 4.5's cross-*configuration* findings (H2's 4 distinct configs vary
declared axes, not RNG draws), but the "n_sequences" and "not driven by
isolated sequences" language should be read as counting configurations,
not independent random samples, for any primary-grid analysis. Flagged
honestly here rather than carried forward silently; not fixed in this
tranche (out of scope -- introducing genuine per-cell randomness for
rare-prevalence cells would require a design change, e.g. jittering
consumption values, deferred to a future tranche if ever needed).

## Baseline integrity

Full worktree suite unchanged since the prior commit (595 passed, 1
skipped, 0 failures with the worktree fix); `src/compitum` remains 100.00%
line+branch covered, mypy `--strict` clean. This tranche's scripts run in
well under a second (pure Python/numpy, no JAX).

## Conclusion for tranche 5

Both open items are now resolved with concrete, actionable findings rather
than left as unexplained anomalies:

1. The slack calibration issue is **partially** a generator artifact (now
   fixed, halves the effect) and **partially** a genuine, unresolved
   property of the frozen pacing controller's short-horizon behavior.
2. The extreme-payoff cliff is **not one thing**: mostly a legitimate,
   maximally-sharp-by-construction pacing benefit or a shared
   environmental threshold, but with at least one genuine, narrow
   controller non-monotonicity mixed in.

Per the user's authorization, FabricPC may now be reintroduced in bounded
residual shadow mode (tranche 5) -- with the explicit caveat that the
residual corrector inherits a base controller that is not perfectly
dormant in slack short-horizon cells and has at least one known narrow
non-monotonic region; the tranche 5 gate's non-inferiority and boundary
-stability criteria inherit these same known imperfections as their
baseline, not a clean reference.
