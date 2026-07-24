"""Tranche 3: an experiment-owned dynamic resource/regret laboratory.

Tranche 2 (`compitum.constraint_oracle`) found that the frozen
`ReflectiveConstraintSolver`'s shared feasibility test does not ordinarily
create a route-specific feasible set, so a present-slack pressure target
carries little temporal information for FabricPC to add to. This package
builds the substrate tranche 2 was missing: routes that genuinely consume
different, cumulative, time-varying resources, so FabricPC can be tested
against the metric that actually matters -- held-out cumulative constrained
regret -- rather than a classification/MAE proxy.

Nothing here touches `constraints.shadow_prices`, `SwitchCertificate`, or
any production routing path. See docs/adr/0003-dynamic-constraint-regret.md.
This package is dependency-free beyond numpy; it never imports FabricPC or
JAX.
"""

from .belief_action_pricing import (
    StepTrace,
    action_shadow_charge,
    run_shadow_charge_policy,
    unit_marginal_prices,
)
from .belief_action_pricing_v2 import run_shadow_charge_policy_v2
from .belief_bellman import BellmanOracle, BellmanStateBudgetExceeded
from .belief_bellman_v2 import BeliefSensitiveBellmanOracle
from .belief_channels import CHANNEL_DIMENSION as BELIEF_CHANNEL_DIMENSION
from .belief_channels import (
    BeliefChannelHistory,
    advance_belief_history,
    compute_belief_channel_vector,
)
from .belief_hmm_filter import belief_high_from_scalar, hmm_filter_step
from .belief_online_optimum import (
    online_optimum_as_hindsight_result,
    run_online_optimal_policy,
)
from .belief_pricing import (
    BeliefEstimator,
    BeliefPricingController,
    ExactBeliefEstimator,
    HmmBeliefEstimator,
    LookupBeliefEstimator,
    RidgeBeliefEstimator,
    build_belief_training_pairs,
)
from .belief_regime import GRID_UNIT as BELIEF_GRID_UNIT
from .belief_regime import MODEL_NAMES as BELIEF_MODEL_NAMES
from .belief_regime import STEPS as BELIEF_STEPS
from .belief_regime import (
    filtered_belief,
    generate_belief_dataset,
    generate_belief_sequence,
    observation_probability,
    predict_belief,
)
from .belief_regime_v2 import (
    U_HIGH_OPPORTUNITY_DEFAULT,
    U_NORMAL_OPPORTUNITY_DEFAULT,
    expected_opportunity_utility,
    generate_belief_dataset_v2,
    generate_belief_sequence_v2,
)
from .channels import CHANNEL_DIMENSION, compute_regret_channel_vector
from .diagnostics import conservation_depletion_split
from .dual_controller import DualController, price_utilities
from .environment import (
    GRID_UNIT,
    MODEL_NAMES,
    RESOURCE_NAMES,
    SCENARIOS,
    DynamicCase,
    DynamicSequence,
    generate_dynamic_dataset,
)
from .forecaster import EWMAForecaster
from .hindsight import HindsightResult, compute_hindsight_optimum
from .metrics import (
    PolicyRunResult,
    bootstrap_ci,
    paired_regret_deltas,
    regret_metrics,
)
from .pricing import (
    PacingController,
    PricingController,
    PricingUpdateContext,
    ReactiveController,
    total_available_over_horizon,
)
from .residual_channels import (
    CHANNEL_DIMENSION as RESIDUAL_CHANNEL_DIMENSION,
)
from .residual_channels import (
    ResidualChannelHistory,
    advance_history,
    compute_residual_channel_vector,
)
from .residual_pricing import ResidualCorrectionRecord, ResidualPricingController
from .residual_target import (
    LambdaInterval,
    compute_oracle_compatible_interval,
    oracle_price_residual,
)
from .scarcity_scenarios import (
    REFERENCE_PARAMS,
    SCARCITY_MODEL_NAMES,
    SCARCITY_RESOURCE_NAMES,
    ScarcityParams,
    build_scarcity_sequence,
    generate_corrected_slack_dataset,
    generate_primary_dataset,
    generate_secondary_dataset,
    primary_grid,
    secondary_sweeps,
)
from .simulator import ForecastContext, PolicyDecision, simulate_policy
from .windowed_predictor import RidgeModel, fit_ridge, flatten_window, predict_ridge

__all__ = [
    "BELIEF_CHANNEL_DIMENSION",
    "BELIEF_GRID_UNIT",
    "BELIEF_MODEL_NAMES",
    "BELIEF_STEPS",
    "CHANNEL_DIMENSION",
    "GRID_UNIT",
    "MODEL_NAMES",
    "REFERENCE_PARAMS",
    "RESIDUAL_CHANNEL_DIMENSION",
    "RESOURCE_NAMES",
    "SCARCITY_MODEL_NAMES",
    "SCARCITY_RESOURCE_NAMES",
    "SCENARIOS",
    "U_HIGH_OPPORTUNITY_DEFAULT",
    "U_NORMAL_OPPORTUNITY_DEFAULT",
    "BeliefChannelHistory",
    "BeliefEstimator",
    "BeliefPricingController",
    "BeliefSensitiveBellmanOracle",
    "BellmanOracle",
    "BellmanStateBudgetExceeded",
    "DualController",
    "DynamicCase",
    "DynamicSequence",
    "EWMAForecaster",
    "ExactBeliefEstimator",
    "ForecastContext",
    "HindsightResult",
    "HmmBeliefEstimator",
    "LambdaInterval",
    "LookupBeliefEstimator",
    "PacingController",
    "PolicyDecision",
    "PolicyRunResult",
    "PricingController",
    "PricingUpdateContext",
    "ReactiveController",
    "ResidualChannelHistory",
    "ResidualCorrectionRecord",
    "ResidualPricingController",
    "RidgeBeliefEstimator",
    "RidgeModel",
    "ScarcityParams",
    "StepTrace",
    "action_shadow_charge",
    "advance_belief_history",
    "advance_history",
    "belief_high_from_scalar",
    "bootstrap_ci",
    "build_belief_training_pairs",
    "build_scarcity_sequence",
    "compute_belief_channel_vector",
    "compute_hindsight_optimum",
    "compute_oracle_compatible_interval",
    "compute_regret_channel_vector",
    "compute_residual_channel_vector",
    "conservation_depletion_split",
    "expected_opportunity_utility",
    "filtered_belief",
    "fit_ridge",
    "flatten_window",
    "generate_belief_dataset",
    "generate_belief_dataset_v2",
    "generate_belief_sequence",
    "generate_belief_sequence_v2",
    "generate_corrected_slack_dataset",
    "generate_dynamic_dataset",
    "generate_primary_dataset",
    "generate_secondary_dataset",
    "hmm_filter_step",
    "observation_probability",
    "online_optimum_as_hindsight_result",
    "oracle_price_residual",
    "paired_regret_deltas",
    "predict_belief",
    "predict_ridge",
    "price_utilities",
    "primary_grid",
    "regret_metrics",
    "run_online_optimal_policy",
    "run_shadow_charge_policy",
    "run_shadow_charge_policy_v2",
    "secondary_sweeps",
    "simulate_policy",
    "total_available_over_horizon",
    "unit_marginal_prices",
]
