"""Oracle-compatible pricing residual target (tranche 5).

Built from the exact hindsight sequence oracle plus each case's already
-declared utility/consumption -- used only to construct OFFLINE training
targets. Never available to any online policy or to
``ResidualPricingController`` at decision time. See
docs/adr/0006-fabricpc-residual-shadow-pricing.md.

``constraints.shadow_prices`` is never read as ground truth here, matching
every prior tranche's naming/authority discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .environment import DynamicCase


@dataclass(frozen=True)
class LambdaInterval:
    """The set of ``lambda >= 0`` (up to ``lambda_max``) for which pricing
    (``base_utility[m] - lambda * consumption[m]``) would select the oracle's
    choice over every other declared model at this step. ``feasible=False``
    means no price in ``[0, lambda_max]`` reproduces the oracle's choice
    (e.g. the oracle deferred, or the choice is strictly dominated) -- in
    that case ``low``/``high`` are not a meaningful interval and must not be
    read as one; a training pipeline must skip such rows, not average them
    into a false scalar."""

    low: float
    high: float
    feasible: bool


def compute_oracle_compatible_interval(
    case: DynamicCase,
    oracle_choice: str,
    resource: str = "budget",
    lambda_max: float = 20.0,
) -> LambdaInterval:
    """Exact, closed-form (piecewise-linear threshold intersection over
    every pairwise model comparison) -- not estimated. Because the pricing
    decision rule is linear in ``lambda`` per model, each pairwise
    comparison contributes at most one threshold, so the feasible set is
    always a single interval (possibly empty, possibly unbounded above)."""
    if oracle_choice == "defer":
        return LambdaInterval(low=0.0, high=lambda_max, feasible=False)

    low = 0.0
    high = lambda_max
    oracle_utility = case.base_utility[oracle_choice]
    oracle_consumption = case.expected_consumption[oracle_choice][resource]

    for model, other_utility in case.base_utility.items():
        if model == oracle_choice:
            continue
        other_consumption = case.expected_consumption[model][resource]
        # Require: oracle_utility - lambda*oracle_consumption >
        #          other_utility  - lambda*other_consumption
        # <=> lambda * (other_consumption - oracle_consumption) > other_utility - oracle_utility
        coefficient = other_consumption - oracle_consumption
        rhs = other_utility - oracle_utility
        if coefficient == 0.0:
            if rhs >= 0.0:
                return LambdaInterval(low=low, high=high, feasible=False)
            continue
        threshold = rhs / coefficient
        if coefficient > 0.0:
            low = max(low, threshold)
        else:
            high = min(high, threshold)

    return LambdaInterval(low=low, high=high, feasible=low <= high)


def oracle_price_residual(interval: LambdaInterval, lambda_base: float) -> Optional[float]:
    """The minimal signed nudge to ``lambda_base`` needed to enter the
    oracle-compatible interval: zero whenever pacing already reproduces the
    oracle's choice (the common, expected case), ``None`` when the interval
    is infeasible (excluded from training, never treated as a zero target)."""
    if not interval.feasible:
        return None
    if lambda_base < interval.low:
        return interval.low - lambda_base
    if lambda_base > interval.high:
        return interval.high - lambda_base
    return 0.0
