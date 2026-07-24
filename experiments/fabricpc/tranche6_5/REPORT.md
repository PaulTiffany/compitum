# Tranche 6.5: Bellman-consistent shadow-price curve — report

Branch `experiment/fabricpc-trajectory-observer`. FabricPC pin unchanged:
v0.3.2 at `32ae295182ab944b8f084abaf4a40da2c50bab5f`. Nothing in this
tranche affects route selection, `main`, `SwitchCertificate`, or
`constraints.shadow_prices`.

## The question this tranche answers

Tranche 6 was accepted as identifying a precise translation failure, not
as closing FabricPC×Compitum: its scalar Bellman marginal price, fed
through the existing linear greedy `utility - lambda*consumption` rule,
did not beat pacing — but the rule itself is only a local, first-order
approximation to a discrete action's actual opportunity cost, invalid
wherever the value function has kinks (which it hand-verifiably does).
This tranche replaces that translation with the exact discrete shadow
charge of each candidate action, computed directly from the same
`BellmanOracle` — no new environment, predictor, or search — and asks
whether *that* correction recovers the value tranche 6 was looking for.
See docs/adr/0008-bellman-consistent-shadow-price-curve.md.

## What was built

- **`src/compitum/regret_lab/belief_action_pricing.py`** —
  `unit_marginal_prices`/`action_shadow_charge`: the exact discrete
  opportunity cost of an action, closed-form (two value-function
  lookups) rather than a linear rate times consumption.
  `run_shadow_charge_policy`: selects
  `argmax[immediate_utility(action) - action_shadow_charge(action)]`,
  provably equivalent (up to an action-independent constant) to full
  Bellman-Q selection. `StepTrace` records the complete belief-timing
  audit trail per step.
- **Belief timing audit, resolved**: the continuation value after an
  action must use `belief_next` (posterior, projected forward one
  transition using this step's own now-observed signal) — not
  `belief_prior` (tranche 6's scalar price's own, differently-justified
  convention) and not the raw, unprojected posterior. Every tranche-6
  `BeliefEstimator` is reused completely unchanged; only what belief is
  fed into the continuation-value lookup, and when, changed.
- **Gate A-prime** (`tests/regret_lab/test_belief_action_pricing.py`):
  with the exact belief, `run_shadow_charge_policy`'s choices are
  required to be bit-identical to `belief_online_optimum.run_online_optimal_policy`'s
  own choices, since this is a mathematical identity, not a statistical
  hope. One sign bug in `unit_marginal_prices`'s negative-`num_units`
  branch was caught and fixed before any test passed — found by the
  telescoping-identity check specifically, not by the exact-equivalence
  check (which would not have caught it, since routing never calls
  `unit_marginal_prices` directly).
- **`experiments/fabricpc/tranche6_5/run_shadow_charge_pilot.py`** — a
  two-phase pilot: Gate A-prime is verified on the real held-out test
  split plus 5 independent robustness seeds *before* any training;
  tranche 6's Part B infrastructure (ridge fit, FabricPC `train_pcn`/
  `train_backprop`, both run once, unchanged) is only exercised after it
  passes.
- 15 new tests, 100% line+branch coverage maintained, mypy `--strict`
  clean, ruff clean (815 project-wide tests passing before this pilot
  run).

## Result 1: Gate A-prime passes perfectly

Zero mismatches between `run_shadow_charge_policy` (exact belief) and
the literal Bellman-optimal online policy, across all 35 held-out test
sequences and all 5 independent robustness seeds (`4242, 1, 2, 3, 100`).
Identical choices, identical cumulative utility, at every single step.
The translation is provably correct, not merely close.

## Result 2: the correction works — decisively

| arm | mean regret (vs. exact online optimum) | mean regret (vs. hindsight) |
| --- | --- | --- |
| no pricing | 2.629 | 3.314 |
| frozen pacing (tranche 6's baseline) | 1.829 | 2.514 |
| exact belief + **scalar** price (tranche 6's failed ablation) | 1.943 | 2.629 |
| exact belief + **shadow charge** (this tranche) | **0.000** | 0.686 |

Exact-belief Bellman pricing, translated through the corrected discrete
shadow charge, achieves **exactly zero regret against the true online
optimum** (guaranteed by Gate A-prime's own identity) and clearly beats
both frozen pacing and tranche 6's scalar-price arm. `recoverable_gap`
(pacing's regret minus the exact-belief-shadow-charge arm's regret) is
**1.829** — substantial, not a rounding artifact. This directly confirms
the diagnosis this tranche was authorized to test: tranche 6's
bottleneck was the linear scalarization, not the environment's
economics and not belief quality.

## Result 3: Gate B passes — the belief is learnable

| predictor | test MSE vs. `belief_prior` | vs. naive constant-mean baseline (0.00609) |
| --- | --- | --- |
| ridge | 0.0000169 | 360x lower |
| FabricPC (backprop) | 0.00129 | 4.7x lower |
| FabricPC (predictive coding) | 0.00127 | 4.8x lower |

All three comfortably clear the recovery threshold (test MSE < 50% of
naive). FabricPC trained genuinely both ways (backprop control and
predictive coding) from identical initial parameters on the same
topology; both converge to essentially the same quality.

## Result 4: Gate C's premise does not apply here — a real, if deflating, finding

**Every arm — HMM, ridge, FabricPC (backprop), FabricPC (predictive
coding), and even the deliberately shuffled FabricPC control — achieves
exactly zero regret, identical route choices, and a 0.0% boundary
-crossing rate against the true online optimum.** This is not a
measurement coincidence. Directly verified by scanning belief across a
9-point grid spanning `[0, 1]` at all 350 (35 sequences x 10 steps)
states actually visited in the test set: **the optimal action never
changes as belief varies, at any visited state.** Once the shadow-charge
correction is applied, the argmax action is fully determined by
`(remaining_steps, budget, observed_opportunity)` alone in this specific
environment's parameterization — belief affects the *value* the Bellman
table assigns (which is why the exact-belief arm's *regret* calculation
is meaningful) but never the *argmax over actions* at a state that is
ever actually reached.

This connects to an empirical finding made while building
`belief_action_pricing.py`: extensive adversarial search (fixed beliefs
at 0.0/0.05/0.95/1.0, oscillating random beliefs, horizons up to 40
steps, hundreds of seeds) never once found a state where a wrong belief
caused the shadow-charge rule to reject an available, affordable,
higher-utility action. That finding generalizes exactly to what this
pilot's boundary-crossing measurement shows directly: "opportunity"'s
flat, non-varying utility (8.0, dominating "spend" at 2.0 and "conserve"
at 1.0) combined with this environment's discrete consumption
granularity means no belief value, right or wrong, ever flips which
action is best.

Consequently, Gate C's literal criteria (beat shuffled, materially
improve on ridge) **fail** — but not because FabricPC underperforms.
They fail because there is no state in this environment where any
belief source could be distinguished from any other by this decision
rule. `captured_fraction = 1.0` for every learned arm (all tied with the
exact-belief ceiling), which is the best possible economic outcome, not
a null result — it just means this specific environment cannot serve as
a further test of belief-estimation *quality* once the scalarization bug
is fixed, since the decision task collapsed to something belief
-invariant.

| gate C criterion | result |
| --- | --- |
| beats frozen pacing | **true** (regret 0.000 vs. 1.829) |
| beats shuffled belief control | false — shuffled also achieves 0.000 regret |
| materially improves on ridge | false — ridge also achieves 0.000 regret |
| non-inferior to backprop control | true (both 0.000) |
| captured_fraction > 0 with CI support | **true** (1.0) |
| `gate_c.passed` | **false** (by the literal AND of all criteria) |

## Interpretation

```text
Gate A-prime: translation is exactly correct.
Gate A economics: the correction recovers the FULL recoverable gap (1.829) -- decisive.
Gate B: belief is learnable, by all three predictors, far beyond a naive baseline.
Gate C: uninformative here, not failing -- this environment has no
        belief-sensitive decision points once pricing is corrected.
```

The tranche's authorizing question -- "does the shadow-charge
correction recover the value tranche 6 was looking for?" -- is answered
**yes, completely**, for the economic/translation half (Gates A-prime
and the recoverable-gap comparison). The belief-estimation half (Gate C)
cannot be further discriminated in this specific environment: any
future work wanting to compare FabricPC against alternatives on belief
-estimation quality specifically needs an environment where *some*
reachable state has a genuinely belief-dependent optimal action (e.g.
opportunity utility that varies by regime rather than being flat, or a
tighter budget/wider utility-gap regime forcing real trade-offs) --
this environment, as parameterized, does not have one.

## Honest methodology notes

- Boundary-sensitive belief diagnostics were narrowed to a
  boundary-crossing rate (fraction of steps where an arm's choice
  differs from the true online optimum's own choice at that state)
  rather than precomputing explicit belief-interval decision boundaries
  per `(time, budget, observation)` state, per the declared runtime
  discipline. In the event, the simpler measure was sufficient to reveal
  the finding above (0% crossings for every arm) -- the heavier
  machinery would not have added information here.
- Arm 5 (true-parameter HMM) was not required to beat arm 4 (exact
  belief); it is reported as an oracle-quality structured ceiling, per
  ADR 0008. In this pilot both achieve identical (zero) regret, for the
  same reason every arm does.
- 50 train / 15 val / 35 test sequences, 3 declared training seeds per
  FabricPC training method, one fixed topology, no hyperparameter search
  -- reusing tranche 6's Part B infrastructure completely unchanged.
  Total pilot runtime: ~142 seconds.

## Preserved conclusions

Tranches 1-6's results stand. Tranche 6's own conclusion is now
sharpened, not overwritten: its scalar-price arm's failure to beat
pacing was specifically a scalarization defect (confirmed by this
tranche's Gate A-prime and recoverable-gap results), not evidence that
exact belief or Bellman pricing lack value in this environment -- they
recover the full gap once correctly translated into an action.
