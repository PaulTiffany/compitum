"""Declared per-step channel for the belief-estimation task (tranche 6).
Dependency-free. FabricPC (and every other predictor tested) observes a
window of these vectors, never a Bellman price, hindsight choice, future
utility, or future realized consumption -- see
docs/adr/0007-belief-state-fabricpc-bellman-pricing.md.

Every field is knowable at or before the current step's decision: the
previous step's chosen route, its realized consumption and utility, the
replenishment observed, remaining resource, time remaining, and this
step's own opportunity-availability signal (already revealed before a
decision is made, per the environment's timing convention -- see
``belief_regime.py``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from .belief_regime import STEPS
from .environment import DynamicCase

CHANNEL_DIMENSION = 11
BUDGET_NORM = 20.0
RECENT_OPPORTUNITY_WINDOW = 5

_ROUTE_INDEX = {"conserve": 0, "spend": 1, "opportunity": 2, "defer": 3}


@dataclass
class BeliefChannelHistory:
    previous_route: Optional[str] = None
    previous_realized_consumption: float = 0.0
    previous_realized_utility: float = 0.0
    previous_replenishment: float = 0.0
    recent_opportunities: Deque[bool] = field(
        default_factory=lambda: deque(maxlen=RECENT_OPPORTUNITY_WINDOW)
    )


def compute_belief_channel_vector(
    remaining: float,
    case: DynamicCase,
    history: BeliefChannelHistory,
    steps_left: int,
    total_steps: int = STEPS,
    resource: str = "budget",
) -> np.ndarray:
    vector = np.zeros(CHANNEL_DIMENSION, dtype=float)
    if history.previous_route is not None:
        vector[_ROUTE_INDEX[history.previous_route]] = 1.0
    vector[4] = history.previous_realized_consumption
    vector[5] = history.previous_realized_utility
    vector[6] = history.previous_replenishment
    vector[7] = remaining / BUDGET_NORM
    vector[8] = steps_left / total_steps if total_steps > 0 else 0.0
    opportunity_now = case.base_utility.get("opportunity", 0.0) > 0.0
    vector[9] = 1.0 if opportunity_now else 0.0
    vector[10] = (
        sum(history.recent_opportunities) / len(history.recent_opportunities)
        if history.recent_opportunities
        else 0.0
    )
    return vector


def advance_belief_history(
    history: BeliefChannelHistory,
    chosen: str,
    case: DynamicCase,
    resource: str = "budget",
) -> BeliefChannelHistory:
    opportunity_now = case.base_utility.get("opportunity", 0.0) > 0.0
    recent = deque(history.recent_opportunities, maxlen=RECENT_OPPORTUNITY_WINDOW)
    recent.append(opportunity_now)
    if chosen == "defer":
        realized_consumption = 0.0
        realized_utility = 0.0
    else:
        realized_consumption = case.realized_consumption[chosen][resource]
        realized_utility = case.base_utility[chosen]
    return BeliefChannelHistory(
        previous_route=chosen,
        previous_realized_consumption=realized_consumption,
        previous_realized_utility=realized_utility,
        previous_replenishment=case.replenishment.get(resource, 0.0),
        recent_opportunities=recent,
    )
