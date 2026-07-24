# ADR 0009: belief-sensitive shadow-charge validation (tranche 7)

Status: accepted, observation-only. Does not modify the shadow-charge
pricing mechanism itself (tranche 6.5, frozen and reused unchanged);
changes only the environment's payoff process, per the authorizing
brief's explicit scope.

## Gate 0 outcome

**Passed, on the second development-grid pass.** The first pass (8
configs varying only `U_NORMAL_OPPORTUNITY`/`U_HIGH_OPPORTUNITY`/initial
budget) found genuine belief-dependent decision boundaries at 56-73 of
77-92 reachable states, but the exact-belief policy tied EXACTLY with
fixed-prior and shuffled belief-blind controls in nearly every config --
diagnosed directly (inspecting real belief trajectories) as an
observation-informativeness artifact: tranche 6's fixed
`P_OPPORTUNITY[HIGH]=0.35` vs `P_OPPORTUNITY[NORMAL]=0.05` (gap 0.30)
makes a single "opportunity available" observation so decisive on its
own (`filtered_belief(0.5, True) = 0.875`) that a naive one-shot read
already lands on the same side of the decision boundary as several steps
of accumulated exact belief, regardless of history.

A second pass, freezing payoff/budget at reasonable values
(`u_normal=1.0`, `u_high=8.0`, `initial_budget=6.0`, the budget value the
first pass's own data showed had far higher boundary occupancy) and
instead varying observation informativeness and regime persistence --
the two additional dimensions the brief itself named -- found a
passing configuration:

```text
p_opportunity_normal=0.15, p_opportunity_high=0.25   (gap narrowed 0.30 -> 0.10)
transition_normal_to_high=0.2, transition_high_to_high=0.6   (unchanged)
```

75.3% of reachable states (58/77) have a genuine belief-dependent action
boundary; 19.3% of steps across 30 diagnostic sequences land within 0.1
belief of one (exceeding the declared 10% minimum); the exact-belief
policy beats fixed-prior, inverted, and shuffled controls in point
-estimate regret (means -0.43, -0.43, -0.80 respectively -- Gate 0 checks
direction, per its own literal criterion, not full statistical
significance, which belongs to the actual pilot's larger held-out test
set); and `exact_mean_regret_vs_online_optimum = 0.0`, confirming the
Gate-A-prime correctness identity still holds exactly at this
configuration. This configuration is now frozen for the ten-arm pilot
(`experiments/fabricpc/tranche7/run_ten_arm_pilot.py`) -- see
`experiments/fabricpc/tranche7/artifacts/gate0_report.json` for the full
grid and selection record.

Two further modules were required by necessity (not initially planned)
once observation/transition tuning proved necessary:
`belief_pricing.ExactBeliefEstimator`/`HmmBeliefEstimator` and
`belief_online_optimum.run_online_optimal_policy` hardcode tranche 6's
fixed parameters internally, so reusing them unchanged with tuned
parameters would have silently made the "exact" belief tracker and the
"exact online optimum" reference both wrong for the environment actually
being evaluated. `ExactBeliefEstimatorV2`/`HmmBeliefEstimatorV2`
(`belief_action_pricing_v2.py`) and `run_online_optimal_policy_v2`/
`online_optimum_as_hindsight_result_v2` (new `belief_online_optimum_v2.py`)
are minimal parameterized siblings, verified via the same
Gate-A-prime-style exact-equivalence check at the tuned parameters (15
seeds) plus independent scalar-vs-matrix cross-validation.

## Governing correction

Tranche 6.5 proved the shadow-charge translation exactly correct (Gate
A-prime) and recovered the full economic gap over pacing -- but also
found, directly (scanning belief across `[0, 1]` at all 350 states
visited in the test set, and independently via the pilot's own 0%
boundary-crossing measurement), that no reachable action in that
environment ever depended on belief: "opportunity" had a single fixed
payoff regardless of the hidden regime, so belief only ever affected a
Bellman *value* used for regret accounting, never an *argmax over
actions*. Every belief source -- exact, learned, even shuffled -- tied
at exactly zero regret. This is benchmark unidentifiability, not
evidence about FabricPC.

## What changed, and what did not

**Changed**: `belief_regime_v2.py`'s "opportunity" payoff now depends on
the hidden regime -- `U_NORMAL_OPPORTUNITY` if the true regime is
NORMAL, `U_HIGH_OPPORTUNITY` if HIGH -- drawn and recorded in
`DynamicCase.base_utility["opportunity"]` as ground truth for
utility/regret accounting, but never revealed to any policy directly.
Every policy, including the exact-belief oracle, can only value taking
"opportunity" via the belief-weighted expectation
`(1 - q) * U_NORMAL_OPPORTUNITY + q * U_HIGH_OPPORTUNITY`
(`expected_opportunity_utility`), using the posterior `q` after this
step's own observation -- never the true realized value.

**Not changed**: the discrete shadow-charge formula
(`action_shadow_charge`/`unit_marginal_prices`, reused byte-for-byte from
`belief_action_pricing.py`), the belief-timing convention (prior ->
posterior -> predicted-next-belief), the FabricPC topology and PC/backprop
training procedures, the window size, optimizer settings, three declared
seeds, ridge/HMM controls, the exact online comparator
(`belief_online_optimum.run_online_optimal_policy`, reused completely
unchanged -- it never scores an action itself, only delegating to
`oracle.best_action_given_observation`, which already performs the
correct belief-weighted computation internally), and every quality-gate
and provenance mechanism from tranches 6/6.5.

`BellmanOracle` (tranche 6/6.5, frozen) is not modified. A new,
structurally-identical class, `BeliefSensitiveBellmanOracle`
(`belief_bellman_v2.py`), differs only in how it values the "opportunity"
branch of its recursion -- same memoization strategy, same method
signatures, so `action_shadow_charge`/`unit_marginal_prices` work with it
unchanged (they only ever call `.value(...)`, never touching a concrete
type). The one genuine adaptation needed is `run_shadow_charge_policy_v2`
(`belief_action_pricing_v2.py`): unlike the exact online optimum, this
function computes each candidate's score itself, so it must be told to
value "opportunity" by the belief-weighted expectation rather than
`case.base_utility["opportunity"]` (the case's true, hidden-regime value)
-- reusing this unchanged would have silently leaked the hidden regime
into the routing decision. Verified directly by a dedicated regression
test: a policy fed a fixed, confidently-wrong belief never selects
"opportunity" even on sequences where the true regime is HIGH and the
true payoff would have been large.

## Gate 0: benchmark identifiability (mandatory before any training)

Before training or evaluating FabricPC: enumerate the exact
Bellman-optimal action over a dense belief grid for every reachable
`(time, remaining budget, observation)` state; confirm at least one
state has two different optimal actions for different beliefs, several
held-out trajectories visit states near such a boundary, the exact
-belief policy beats fixed-prior/shuffled/inverted-belief controls, the
advantage is not produced by one isolated sequence, and belief
perturbations across a boundary flip the selected action exactly as
predicted. See `experiments/fabricpc/tranche7/run_gate0_identifiability.py`
for the implementation and the frozen, boundary-occupancy-selected
environment configuration.

## Required arms

No pricing; frozen pacing; exact belief + shadow charge; fixed-prior
belief + shadow charge; true-parameter HMM + shadow charge; ridge belief
+ shadow charge; same topology trained by backprop + shadow charge; same
topology trained by FabricPC predictive coding + shadow charge; shuffled
FabricPC belief + shadow charge; inverted/deliberately-biased belief +
shadow charge.

## Primary economics gate

Regret vs. the exact information-matched online optimum remains primary.
FabricPC passes only if it beats frozen pacing, beats fixed-prior and
shuffled-belief controls, has zero additional violations, captures a
positive statistically-supported fraction of the recoverable gap, is
non-inferior to the same topology trained by backprop, behaves better
near belief-decision boundaries than shuffled belief, and remains useful
after latency.

## Boundary-sensitive diagnostics

For every visited decision state: exact belief, predicted belief,
nearest Bellman decision-boundary belief, distance to boundary, exact
optimal action, predicted-belief action, whether the prediction crossed
to the wrong side, and instantaneous/downstream regret caused by any
crossing -- reported alongside overall MSE/calibration/Brier/log-loss,
since a lower global MSE is secondary to whether errors cross
economically consequential boundaries.

## Stop boundary

Complete locally: the belief-sensitive environment; Gate 0's
decision-boundary proof and frozen configuration; the ten-arm pilot;
exact checkpoint/artifact hashes; a regret and boundary-sensitive report;
a final synthesis of tranches 1-7 and a merge-candidate inventory. Stop
before: production integration, changes to `constraints.shadow_prices` or
`SwitchCertificate`, push, PR, tag, release, or wiki/paper updates.
