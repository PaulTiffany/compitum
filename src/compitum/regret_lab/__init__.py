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
from .simulator import ForecastContext, PolicyDecision, simulate_policy

__all__ = [
    "CHANNEL_DIMENSION",
    "GRID_UNIT",
    "MODEL_NAMES",
    "RESOURCE_NAMES",
    "SCENARIOS",
    "DualController",
    "DynamicCase",
    "DynamicSequence",
    "EWMAForecaster",
    "ForecastContext",
    "HindsightResult",
    "PolicyDecision",
    "PolicyRunResult",
    "bootstrap_ci",
    "compute_hindsight_optimum",
    "compute_regret_channel_vector",
    "generate_dynamic_dataset",
    "paired_regret_deltas",
    "price_utilities",
    "regret_metrics",
    "simulate_policy",
]
