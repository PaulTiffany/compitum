"""Declared per-step channel for the residual-pricing correction task
(tranche 5). Dependency-free. A genuine multi-step WINDOW of these
vectors (not one static snapshot) is what FabricPC observes -- see
docs/adr/0006-fabricpc-residual-shadow-pricing.md.

Explicitly excluded from every field: future utilities, future realized
consumption, future opportunity arrival, hindsight choices, evaluation
labels. Every input here is knowable at or before decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .environment import DynamicCase

CHANNEL_DIMENSION = 13
BUDGET_NORM = 20.0
OPPORTUNITY_CONSUMPTION_CLIP = 50.0
OPPORTUNITY_SEEN_CAP = 20


@dataclass
class ResidualChannelHistory:
    """Bookkeeping a caller (``ResidualPricingController``) maintains
    across steps and feeds back in; kept separate from the pure channel
    -computation function below so that function stays trivially testable."""

    previous_lambda: float = 0.0
    previous_route: Optional[str] = None
    route_before_previous: Optional[str] = None
    steps_since_opportunity_seen: int = OPPORTUNITY_SEEN_CAP
    last_forecast_error: float = 0.0


def compute_residual_channel_vector(
    remaining: float,
    case: DynamicCase,
    lambda_base: float,
    pacing_error: float,
    history: ResidualChannelHistory,
    steps_left: int,
    total_steps: int,
    resource: str = "budget",
) -> np.ndarray:
    """One step's declared observation. ``pacing_error`` is the frozen
    pacing controller's own cumulative-usage-vs-target error (its internal
    bookkeeping, not re-derived here) -- the single richest scalar summary
    of "is scarcity currently consequential" the base controller has."""
    vector = np.zeros(CHANNEL_DIMENSION, dtype=float)
    vector[0] = remaining / BUDGET_NORM
    vector[1] = pacing_error
    vector[2] = case.replenishment.get(resource, 0.0)
    vector[3] = case.expected_consumption["conserve"][resource]
    vector[4] = case.expected_consumption["spend"][resource]
    vector[5] = min(
        case.expected_consumption["opportunity"][resource], OPPORTUNITY_CONSUMPTION_CLIP
    )

    priced = {
        m: case.base_utility[m] - lambda_base * case.expected_consumption[m][resource]
        for m in case.base_utility
    }
    ranked = sorted(priced.values(), reverse=True)
    vector[6] = ranked[0] - ranked[1] if len(ranked) > 1 else 0.0
    vector[7] = lambda_base
    vector[8] = lambda_base - history.previous_lambda
    vector[9] = (
        1.0
        if history.previous_route is not None
        and history.route_before_previous is not None
        and history.previous_route != history.route_before_previous
        else 0.0
    )
    vector[10] = steps_left / total_steps if total_steps > 0 else 0.0
    vector[11] = (
        min(history.steps_since_opportunity_seen, OPPORTUNITY_SEEN_CAP) / OPPORTUNITY_SEEN_CAP
    )
    vector[12] = history.last_forecast_error
    return vector


def advance_history(
    history: ResidualChannelHistory,
    case: DynamicCase,
    chosen: str,
    lambda_base: float,
    forecast_error: float = 0.0,
) -> ResidualChannelHistory:
    """Returns the history state for the *next* step, given what happened
    this step."""
    opportunity_seen_now = case.base_utility.get("opportunity", 0.0) > 0.0
    steps_since = (
        0
        if opportunity_seen_now
        else min(history.steps_since_opportunity_seen + 1, OPPORTUNITY_SEEN_CAP)
    )
    return ResidualChannelHistory(
        previous_lambda=lambda_base,
        previous_route=chosen,
        route_before_previous=history.previous_route,
        steps_since_opportunity_seen=steps_since,
        last_forecast_error=forecast_error,
    )
