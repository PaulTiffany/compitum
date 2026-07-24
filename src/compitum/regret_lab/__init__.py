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

__all__ = [
    "CHANNEL_DIMENSION",
    "GRID_UNIT",
    "MODEL_NAMES",
    "REFERENCE_PARAMS",
    "RESIDUAL_CHANNEL_DIMENSION",
    "RESOURCE_NAMES",
    "SCARCITY_MODEL_NAMES",
    "SCARCITY_RESOURCE_NAMES",
    "SCENARIOS",
    "DualController",
    "DynamicCase",
    "DynamicSequence",
    "EWMAForecaster",
    "ForecastContext",
    "HindsightResult",
    "LambdaInterval",
    "PacingController",
    "PolicyDecision",
    "PolicyRunResult",
    "PricingController",
    "PricingUpdateContext",
    "ReactiveController",
    "ResidualChannelHistory",
    "ResidualCorrectionRecord",
    "ResidualPricingController",
    "ScarcityParams",
    "advance_history",
    "bootstrap_ci",
    "build_scarcity_sequence",
    "compute_hindsight_optimum",
    "compute_oracle_compatible_interval",
    "compute_regret_channel_vector",
    "compute_residual_channel_vector",
    "conservation_depletion_split",
    "generate_corrected_slack_dataset",
    "generate_dynamic_dataset",
    "generate_primary_dataset",
    "generate_secondary_dataset",
    "oracle_price_residual",
    "paired_regret_deltas",
    "price_utilities",
    "primary_grid",
    "regret_metrics",
    "secondary_sweeps",
    "simulate_policy",
    "total_available_over_horizon",
]
