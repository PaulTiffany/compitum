# ADR 0006: FabricPC residual shadow pricing (tranche 5)

Status: accepted, observation-only. Supersedes nothing in ADR 0001-0005;
follows from tranche 4.6's resolution of tranche 4.5's open items.

## Governing rule

> The deterministic pacing controller supplies the price. FabricPC may
> earn the right to supply a bounded, prospective correction -- but only
> if that correction reduces constrained regret beyond both pacing and an
> ordinary sequential predictor.

FabricPC is reintroduced strictly as a prospective observer of the
**pricing residual**, never as an autonomous economic controller and never
as a replacement for the frozen pacing controller established in tranches
4/4.6. Per tranche 4.6: that frozen controller is not a clean reference --
it carries a known, only-partially-understood non-dormancy in short
-horizon slack cells and at least one known narrow non-monotonic region.
Tranche 5's gate criteria are evaluated against that real, imperfect
baseline, not an idealized one.

## Bounded residual architecture

```text
lambda_base[t]        = frozen_pacing_controller(state_history)   # unchanged from tranche 4
delta_lambda_pc[t]     = FabricPC's predicted correction (bounded)
lambda_effective[t]    = clip(lambda_base[t] + gate[t] * delta_lambda_pc[t], 0, lambda_max)
```

`gate[t]` is a boolean (or scalar 0/1) restricting when the correction may
apply at all, per the regional-scope requirement below. `delta_lambda_pc`
always has an explicit maximum magnitude (`max_correction_magnitude`,
declared per experiment, not learned) and a **deterministic fallback of
zero** on any observer failure -- a failed or refused FabricPC observation
never blocks a decision, it just leaves `lambda_effective = lambda_base`.

`src/compitum/regret_lab/residual_pricing.py`'s `ResidualPricingController`
implements this: it wraps a frozen `PacingController` (the tranche-4/4.6
base, completely unmodified), maintains its own bounded window of declared
per-step channel vectors, and exposes the same `lambda_price`/`update`
interface every other `PricingController` does -- `simulate_policy` needs
no further changes. Every step's correction attempt produces a
`ResidualCorrectionRecord` (status one of `applied`, `zero_gate`,
`clipped`, `refused`, `failed`; the raw and clipped correction; the
window size used) -- a lightweight, inspectable provenance trail per step,
analogous in spirit to tranches 1-3's governed `TrajectoryEvidence`
status vocabulary, without depending on that module's FabricPC-observation
-specific raw schema.

**No authority to bypass feasibility**: `lambda_effective` only ever feeds
into the existing, unchanged `price_utilities` function -- it can change
which feasible model looks cheapest, never which models are feasible in
the first place. **No production wiring**: this entire mechanism lives in
`compitum.regret_lab` and `experiments/fabricpc/tranche5/`; nothing here
touches `constraints.shadow_prices` or any file outside this experimental
path.

## Residual target: oracle-compatible lambda interval

Built from the exact hindsight sequence oracle plus each case's realized
outcome -- **only used to construct offline training targets**, never
available to any online policy or the residual controller itself at
decision time. For a single-resource step with oracle choice `c`:

```text
oracle_compatible_interval[t] = { lambda >= 0 :
    argmax_m (base_utility[m] - lambda * consumption[m]) == c
    over every model declared at that step }
```

This is computed exactly (piecewise-linear threshold intersection over all
pairwise model comparisons, `src/compitum/regret_lab/residual_target.py`),
not estimated -- for 3 models this is a small, closed-form interval
(possibly unbounded on one side, possibly empty if no price could ever
reproduce the oracle's choice, e.g. when the oracle deferred). Because this
interval, not a point, is the real ground truth, the **training target is
the minimal signed nudge needed to enter it**:

```text
oracle_price_residual[t] = 0                          if lambda_base[t] in [low, high]
                          = low - lambda_base[t]       if lambda_base[t] < low
                          = high - lambda_base[t]       if lambda_base[t] > high
                          = undefined (excluded)         if the interval is empty (infeasible)
```

This is well-defined, interpretable (a genuine zero target whenever pacing
already reproduces the oracle's choice -- expected to be the common case,
consistent with the gate mostly staying closed), and does not invent false
scalar precision where only an interval exists. `constraints.shadow_prices`
is never read as ground truth here, consistent with every prior tranche.

## FabricPC inputs: a genuine multi-step window

Unlike tranches 1-3 (a single static channel vector per call) and tranche
3's regret-lab channel (also single-step), tranche 5's declared channel
(`src/compitum/regret_lab/residual_channels.py`) is accumulated over a
bounded window of the **last `W` environment steps**, each containing:
remaining resources; cumulative usage relative to the pacing target;
replenishment; per-model expected consumption; realized consumption
revealed so far (respecting `revelation_delay`); utility gaps; current
pacing price (`lambda_base`); recent price changes; recently selected
routes; forecast errors observed so far; steps until horizon end; and a
declared (non-privileged) proxy for uncertainty around future opportunity
arrival (e.g. time since the last opportunity window, if any, within the
observed history -- never the true future arrival time). FabricPC's own
inner settling trajectory (within one call) remains observable and
auditable exactly as before; what's new is that the *input* to that
settling process is now a flattened multi-step window, not one snapshot.

**Explicitly excluded from every input channel:** future utilities, future
realized consumption, future opportunity arrival, hindsight choices, and
any evaluation label. The window only ever contains information available
at or before decision time `t`.

## Required arms (paired, held-out, corrected stable dataset)

1. no pricing;
2. frozen pacing (tranche 4/4.6, unmodified);
3. frozen pacing + a simple non-FabricPC windowed predictor (ridge
   regression over the same declared window, fit offline on training
   sequences, frozen at test time -- directly parallel to tranche 3's EWMA
   baseline, but windowed);
4. frozen pacing + FabricPC terminal-state residual (single most-recent
   step only, an ablation isolating whether the window helps at all);
5. frozen pacing + FabricPC trajectory residual (the full declared window);
6. frozen pacing + shuffled/sequence-mismatched FabricPC residual
   (negative control);
7. frozen pacing + FabricPC residual with the gate forcibly held open
   everywhere (an ablation testing whether the regional-scope restriction
   is genuinely protective, not merely inert).

The reactive controller (tranche 3) is retained as historical context in
the report only -- it is not a baseline arm 5 must beat.

## Gate (regret remains primary; do not gate on residual MAE)

Arm 5 (trajectory residual) passes only if it: (1) reduces paired held-out
regret vs. frozen pacing; (2) reduces regret vs. the non-FabricPC windowed
predictor; (3) is significantly better than shuffled trajectories; (4)
adds no additional hard violations; (5) is non-inferior in the corrected
slack and false-scarcity regions from tranche 4.6; (6) does not worsen the
extreme-payoff boundary cliff; (7) is not driven by a few isolated
sequences; (8) remains useful after accounting for observer latency cost.
A numerically inaccurate residual can still cross the correct route
-selection boundary, and a low-MAE residual can still cross the wrong one
-- MAE is reported as a diagnostic, never the gate.

## Regional scope, not a general claim

The phase study (tranche 4.5) supports testing pacing/FabricPC primarily
in longer-horizon, moderate-to-high-consequence, genuinely scarce regimes.
No claim of general pricing improvement is made outside that region. The
correction gate (`gate[t]`) may legitimately stay closed everywhere else
during this first pilot; arm 7 (gate forced open) exists specifically to
test whether that restriction is protective or merely inert.

## Outcome (2026-07-24)

The seven-arm paired shadow pilot (`experiments/fabricpc/tranche5/`) ran on
the 72-cell primary grid (48 train / 24 test cells, split by cell index
since tranche 4.6 found rare-prevalence cells never consume their RNG --
1 sequence per cell, no manufactured duplicate "seeds"). 410 feasible
oracle-compatible training rows out of 576 possible; 1640 real FabricPC
observations, 0 governed failures, p50/p95/max latency 0.49s/0.66s/2.24s.
Full detail in `experiments/fabricpc/tranche5/REPORT.md`. Summary:

**FabricPC's correction is genuinely inert, not harmful.** Arms 4, 5, and 7
(FabricPC terminal-state, full-trajectory, and gate-forced-open residuals)
all produce **byte-identical mean regret to frozen pacing alone** (2.0833),
despite computing and applying real, nonzero corrections every gated step
(mean absolute correction 0.34-0.35, 0% clipped, 0% failed). Route
disagreement vs. frozen pacing is only 1.4% -- the correction exists but
essentially never crosses a decision-relevant threshold. **Gate fails
cleanly: `beats_frozen_pacing: false`** (paired delta exactly 0.0).

**In contrast, the two arms whose corrections *do* cross decision
boundaries more often (5.2% disagreement each) make things worse, not
better:** the non-FabricPC windowed ridge (arm 3, mean regret 2.125,
58% of corrections clipped at the bound -- a poorly-calibrated raw
prediction the magnitude cap is mostly there to contain) and the
shuffled-trajectory control (arm 6, mean regret 2.125). Arm 5 technically
"beats" both of these worse-than-baseline arms by an identical small
margin, but the CI does not exclude zero and, more importantly, beating an
arm that itself underperforms the baseline is not a meaningful bar --
the real failure is arm 5 not beating frozen pacing directly. The one gate
criterion satisfied is `no_additional_violations: true`.

**No claim of improvement is made.** Per the governing rule, FabricPC has
not earned the right to supply a production-relevant correction in this
pilot. No learned predictor is activated; the frozen pacing controller
alone remains the only pricing mechanism with any demonstrated (tranche 4
/4.5/4.6-scoped) benefit.

## Stop boundary

Complete locally: this ADR; the windowed residual-target/channel
machinery; the bounded residual adapter; the non-FabricPC windowed
comparator; the JAX-side windowed FabricPC observer; the seven-arm paired
shadow pilot; a regret-centered report with every required diagnostic;
full tests/coverage/typing/lint. Stop before: production integration,
changes to `constraints.shadow_prices`, `SwitchCertificate` changes, live
route changes, push, PR, or wiki/paper updates.
