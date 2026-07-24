# Module map

Concept-to-canonical-source-path index, at frozen commit
`617f8979daa921d326301266e55740c0746ab95c` (tag
`fabricpc-compitum-shadow-pricing-v1`), branch
`experiment/fabricpc-trajectory-observer`. All paths are repository
-relative. This is an index, not a copy — nothing here is duplicated
into this capsule.

## Bellman continuation value

- `src/compitum/regret_lab/belief_bellman.py` — `BellmanOracle` (tranche
  6/6.5's frozen environment: flat "opportunity" payoff).
- `src/compitum/regret_lab/belief_bellman_v2.py` — `BeliefSensitiveBellmanOracle`
  (tranche 7's belief-sensitive environment: regime-dependent
  "opportunity" payoff via the belief-weighted expectation).

## Discrete unit marginal-price curve

- `src/compitum/regret_lab/belief_action_pricing.py` — `unit_marginal_prices`,
  the telescoping sequence of exact unit marginal prices (used for the
  Gate A-prime correctness check, not for routing directly).

## Action shadow charge

- `src/compitum/regret_lab/belief_action_pricing.py` — `action_shadow_charge`,
  `run_shadow_charge_policy`, `StepTrace` (the closed-form charge and the
  routing loop for tranche 6.5's environment).
- `src/compitum/regret_lab/belief_action_pricing_v2.py` — `run_shadow_charge_policy_v2`,
  `ExactBeliefEstimatorV2`, `HmmBeliefEstimatorV2` (tranche 7's
  belief-sensitive environment; reuses `action_shadow_charge`/
  `unit_marginal_prices`/`StepTrace` from the module above unchanged).

## Online optimum (principal regret comparator)

- `src/compitum/regret_lab/belief_online_optimum.py` — `run_online_optimal_policy`,
  `online_optimum_as_hindsight_result` (tranche 6/6.5's environment).
- `src/compitum/regret_lab/belief_online_optimum_v2.py` — `run_online_optimal_policy_v2`,
  `online_optimum_as_hindsight_result_v2` (tranche 7's environment; a
  parameterized sibling required once Gate 0 needed tuned transition
  /observation parameters).

## Belief-sensitive environment

- `src/compitum/regret_lab/belief_regime.py` — the original hidden-regime
  environment (flat "opportunity" payoff; tranches 6/6.5).
- `src/compitum/regret_lab/belief_regime_v2.py` — the belief-sensitive
  environment (`expected_opportunity_utility`, `filtered_belief_v2`,
  `predict_belief_v2`, `observation_probability_v2`; tranche 7).
- `src/compitum/regret_lab/belief_pricing.py` — `BeliefPricingController`,
  the original (tranche 6) scalar-price arm, retained as the negative
  -translation ablation in the tranche 6.5/7 pilots.
- `experiments/fabricpc/tranche7/run_gate0_identifiability.py` — the
  benchmark-identifiability check and the tiny development grid that
  selected tranche 7's frozen environment configuration.

## Ridge estimator

- `src/compitum/regret_lab/belief_pricing.py` — `RidgeBeliefEstimator`.
- `src/compitum/regret_lab/windowed_predictor.py` — `fit_ridge`,
  `predict_ridge`, the underlying ridge-regression implementation.

## FabricPC PCN estimator

- `experiments/fabricpc/tranche6/fabricpc_belief_model.py` — `train_belief_model("pcn", ...)`,
  `predict_belief_batch`, `FabricPCBeliefEstimator` (genuine predictive
  -coding training via FabricPC's `train_pcn`, on the declared belief
  -regression topology).

## FabricPC backprop control

- `experiments/fabricpc/tranche6/fabricpc_belief_model.py` — `train_belief_model("backprop", ...)`,
  the required same-topology backprop control, trained via FabricPC's
  `train_backprop` from identical initial parameters to the PCN arm.

## Reports and JSON artifacts

- `experiments/fabricpc/FINAL_SYNTHESIS.md` — cross-tranche synthesis.
- `experiments/fabricpc/tranche6_5/REPORT.md`,
  `experiments/fabricpc/tranche6_5/artifacts/shadow_charge_pilot_report.json`
- `experiments/fabricpc/tranche7/REPORT.md`,
  `experiments/fabricpc/tranche7/artifacts/gate0_report.json`,
  `experiments/fabricpc/tranche7/artifacts/ten_arm_pilot_report.json`
- `docs/adr/0008-bellman-consistent-shadow-price-curve.md`,
  `docs/adr/0009-belief-sensitive-shadow-charge-validation.md`

## Tests

- `tests/regret_lab/test_belief_action_pricing.py`,
  `tests/regret_lab/test_belief_action_pricing_v2.py` — Gate A-prime
  exact-equivalence and telescoping-identity tests.
- `tests/regret_lab/test_belief_online_optimum.py`,
  `tests/regret_lab/test_belief_online_optimum_v2.py`
- `tests/regret_lab/test_belief_bellman.py`,
  `tests/regret_lab/test_belief_bellman_v2.py`
- `tests/regret_lab/test_belief_regime.py`,
  `tests/regret_lab/test_belief_regime_v2.py`

## Dependency provenance

- `experiments/fabricpc/fabricpc_install_receipt.json` — pinned FabricPC
  commit, environment, and verification record.
