# Tranche 7: belief-sensitive shadow-charge validation — report

Branch `experiment/fabricpc-trajectory-observer`. FabricPC pin unchanged:
v0.3.2 at `32ae295182ab944b8f084abaf4a40da2c50bab5f`. Nothing in this
tranche affects route selection, `main`, `SwitchCertificate`, or
`constraints.shadow_prices`.

## The question this tranche answers

Tranche 6.5 proved the shadow-charge translation exactly correct and
recovered the full economic gap over pacing — but also found, directly,
that no reachable action in that environment ever depended on belief:
every arm, including a deliberately shuffled FabricPC control, tied at
exactly zero regret. This tranche makes belief genuinely decision
-relevant (Gate 0), then asks the question tranche 6.5 could not: does
trained FabricPC belief inference actually improve constrained routing
regret once belief has real economic consequences? See
docs/adr/0009-belief-sensitive-shadow-charge-validation.md.

## What was built

- **`src/compitum/regret_lab/belief_regime_v2.py`** — regime-dependent
  "opportunity" payoff (`U_NORMAL_OPPORTUNITY`/`U_HIGH_OPPORTUNITY`,
  drawn from the true hidden regime for ground-truth accounting only,
  never revealed directly) plus parameterized observation/transition
  dynamics (`filtered_belief_v2`/`predict_belief_v2`/`observation_probability_v2`),
  needed once Gate 0's own diagnostic showed payoff tuning alone was
  insufficient.
- **`src/compitum/regret_lab/belief_bellman_v2.py`** — `BeliefSensitiveBellmanOracle`,
  structurally identical to tranche 6/6.5's frozen `BellmanOracle`,
  differing only in how it values "opportunity" (the belief-weighted
  expectation, not a fixed constant).
- **`src/compitum/regret_lab/belief_action_pricing_v2.py`** —
  `run_shadow_charge_policy_v2` (scores "opportunity" by belief-weighted
  expectation, never the case's true realized value — verified by a
  dedicated regression test), plus `ExactBeliefEstimatorV2`/`HmmBeliefEstimatorV2`,
  minimal parameterized siblings required once observation/transition
  tuning proved necessary (the originals hardcode tranche 6's fixed
  values internally).
- **`src/compitum/regret_lab/belief_online_optimum_v2.py`** —
  `run_online_optimal_policy_v2`, the same necessity: the frozen
  `run_online_optimal_policy` also hardcodes fixed parameters for its own
  belief tracking, which would have silently made the "exact online
  optimum" reference wrong once those parameters were tuned.
- **`experiments/fabricpc/tranche7/run_gate0_identifiability.py`** — Gate
  0's benchmark-identifiability check, run twice. First pass (payoff
  /budget tuning only) found genuine boundary states but the exact
  -belief policy tied EXACTLY with belief-blind controls — diagnosed
  directly as an observation-informativeness artifact (a single
  "opportunity available" signal was decisive enough on its own to swamp
  any prior). Second pass narrowed the observation gap
  (`P_OPPORTUNITY[NORMAL]=0.15`, `P_OPPORTUNITY[HIGH]=0.25`, vs. tranche
  6's `0.05`/`0.35`) and passed: 75.3% of reachable states have a genuine
  belief-dependent boundary, 19.3% of steps land within 0.1 belief of
  one, and the exact-belief policy beats fixed-prior/inverted/shuffled
  controls in point-estimate regret.
- **`experiments/fabricpc/tranche7/run_ten_arm_pilot.py`** — the frozen
  Gate 0 configuration's ten-arm pilot, reusing tranche 6.5's Part B
  training infrastructure (ridge, FabricPC `train_pcn`/`train_backprop`)
  completely unchanged.
- 72 new regret_lab tests (this tranche alone), 100% coverage maintained,
  mypy `--strict` clean, ruff clean; 905 tests pass project-wide before
  this pilot run.

## Result: economics and learnability are both real; FabricPC specifically underperforms ridge

| arm | mean regret (vs. exact online optimum) | belief test MSE | 
| --- | --- | --- |
| no pricing | 0.371 | — |
| frozen pacing | 0.371 | — |
| exact belief + shadow charge | **0.000** | 0 (ground truth) |
| true-parameter HMM + shadow charge | **0.000** | ~0 (mathematically ~exact) |
| **ridge + shadow charge** | **0.000** | **5.1e-7** |
| fixed-prior belief + shadow charge | 0.057 | — |
| inverted belief + shadow charge | 0.057 | — |
| FabricPC (backprop) + shadow charge | 0.314 | 3.2e-4 |
| FabricPC (predictive coding) + shadow charge | 0.314 | 3.3e-4 |
| shuffled FabricPC belief + shadow charge | 0.457 | — |

**Gate 0 and the economic mechanism both work as intended.** Exact
belief clearly beats every belief-blind control (fixed-prior, inverted,
shuffled), confirming belief now has genuine decision value in this
environment (unlike tranche 6.5's). `recoverable_gap` (pacing regret
minus exact-belief regret) is 0.371 — small in absolute terms (this is a
narrow, bounded pilot) but real and reproducible.

**Ridge — a plain linear regression — fully captures the belief
-estimation task**, achieving belief test MSE of 5.1e-7 (essentially
exact) and, as a direct consequence, **exactly zero regret**, tied with
the true exact-belief oracle and the true-parameter HMM filter. This
conclusively demonstrates that Gate A's exact-belief advantage is
achievable by an ordinary learned model, not just the oracle itself —
the belief-estimation task, as declared, is genuinely learnable and
economically worth solving.

**FabricPC does not reach that quality, under either training rule.**
`fabricpc_backprop` and `fabricpc_pcn` are tied EXACTLY with each other
(0.314 mean regret, identical to the decimal; zero variance in their
paired difference across all 35 test sequences) and both underperform
ridge substantially: their belief test MSE (~3.2-3.3e-4) is roughly
**600x worse than ridge's** (5.1e-7), despite both nominally "recovering"
per Gate B's naive-baseline threshold. This quality gap translates
directly into regret: FabricPC captures only **15.4%** of the total
recoverable gap over pacing (`captured_fraction = 0.154`), leaving 84.6%
on the table that ridge alone recovers in full. FabricPC does not
significantly beat frozen pacing (mean delta -0.057, CI `[-1.257,
1.057]`, straddles zero), does not beat the shuffled-belief negative
control significantly (mean -0.143, CI `[-0.914, 0.714]`), and is
actually *worse* than the fixed-prior control in point-estimate regret
(mean +0.257, CI `[-0.229, 0.857]`) — though none of these differences
reach statistical significance at this sample size, none of them show
FabricPC clearly winning either.

**Primary economics gate: FAILED.** Criteria met: no additional
violations; captured fraction positive; non-inferior to backprop
(trivially, since they tie exactly). Criteria not met: does not
significantly beat pacing; does not significantly beat fixed-prior or
shuffled controls. `gate_economics.passed = false`.

## Interpretation

None of the ADR's four pre-declared interpretive buckets precisely fits
what happened, and forcing the result into one would overstate or
understate it:

- *"Exact belief does not beat belief-blind controls"* — false; it
  clearly does (0.000 vs. 0.057/0.057/0.457).
- *"Exact belief helps, but no learned model does"* — false; ridge
  fully recovers the exact-belief result.
- *"Backprop helps, FabricPC does not"* — false; backprop and
  predictive-coding training are tied exactly, neither helping much.
- *"FabricPC beats pacing and shuffled belief"* — false; neither margin
  is significant, and FabricPC loses to fixed-prior in point estimate.

The actual, precise finding: **this is an architecture/representation
bottleneck specific to FabricPC's learned graph, not a "which training
rule" question and not a "is the task learnable" question.** Both
learning rules converge (val MSE stable across 3 seeds, ~2.6e-4 to
6.5e-4) to a representation that is real and better than the naive
constant baseline, but is roughly 600x less accurate than what a plain
ridge regression achieves on the identical declared window features.
Predictive coding and backprop are not meaningfully distinguishable from
each other here (consistent with prior literature that small, simple
regression tasks often show little difference between the two learning
rules) — the distinguishing factor is FabricPC's small fixed topology
and/or training budget against this specific task, not the choice
between local predictive-coding gradients and global backprop.

## Honest methodology notes

- Gate 0's occupancy/margin criteria were checked by point-estimate
  direction (`mean < 0`), not full statistical significance, per Gate
  0's own literal wording ("has lower regret than") and its role as a
  bounded 30-sequence feasibility screen — the *pilot's* own gates
  (task 7.3) correctly apply full CI-based significance, and it is
  exactly there that FabricPC's advantages fail to clear the bar.
- `fixed_prior` and `inverted_belief` land at the identical regret value
  (0.057) in this specific run — plausible given both are "wrong in a
  structured way" relative to the same exact-belief trajectory (fixed
  -prior ignores history; inverted actively reverses it), but not
  independently re-verified beyond this single test-set realization,
  since regret ties are not, by themselves, evidence of a shared
  mechanism.
- No hyperparameter search, additional training seeds, or architecture
  changes were attempted to close FabricPC's gap with ridge, per the
  runtime-discipline mandate ("no more pricing changes... no
  architecture or hyperparameter search"). Whether a larger/differently
  -tuned FabricPC topology would close this gap is an open question this
  tranche deliberately did not chase.
- 50 train / 15 val / 35 test sequences, 3 declared training seeds per
  FabricPC method, one fixed topology (unchanged from tranche 6/6.5),
  reusing Part B's training infrastructure completely unchanged. Total
  pilot runtime: ~60 seconds.

## Preserved conclusions

Tranches 1–6.5's results stand. Tranche 6.5's own finding (the
shadow-charge correction is exact and economically decisive) is
confirmed again here, in a genuinely belief-sensitive environment: exact
belief clearly beats belief-blind controls, and `recoverable_gap` is
real and positive. What tranche 6.5 could not test — whether trained
FabricPC belief inference converts that opportunity into actual regret
reduction — is now answered: not at this topology, training budget, and
seed count, against a ridge baseline that fully solves the same task.
