"""Pluggable non-learned pricing controllers (tranche 4).

Tranche 3 found that the reactive dual controller already loses to no
pricing at all, and that layering a learned predictor on top only made
regret worse (premature conservation, not genuine improvement). Per
docs/adr/0004-pricing-controller-repair.md, this module establishes and
compares several NON-LEARNED pricing policies before any learned predictor
is reintroduced.

Every controller here exposes the same two things ``simulate_policy``
needs: a ``lambda_price`` dict (fed into the existing, unchanged
``price_utilities``) and an ``update(context)`` method. ``ReactiveController``
wraps the existing, byte-for-byte-unchanged ``DualController`` from
tranche 3, reproducing its exact utilization-error formula so that arm's
behavior is verified identical, not merely similar. ``PacingController`` is
one flexible class realizing plain pacing, pacing+hysteresis, asymmetric
relaxation, and EMA-smoothed/bounded variants via parameters (see the ADR
for why one parameterized class is used instead of four near-duplicates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

from .dual_controller import DualController
from .environment import DynamicCase, DynamicSequence


@dataclass
class PricingUpdateContext:
    """Raw per-step ingredients every controller variant needs -- not a
    single pre-derived error number, since pacing-family controllers need
    cumulative history the old single-step reactive formula discarded.

    ``case`` and ``chosen`` are optional (default ``None``) and unused by
    ``ReactiveController``/``PacingController``; tranche 5's
    ``ResidualPricingController`` uses them to build its declared
    multi-step channel window, which needs every model's utility/
    consumption at this step, not just the chosen model's reservation."""

    resource_names: Tuple[str, ...]
    reservation: Dict[str, float]
    remaining_before: Dict[str, float]
    remaining_after: Dict[str, float]
    step: int
    total_steps: int
    case: Optional[DynamicCase] = None
    chosen: Optional[str] = None


class PricingController(Protocol):
    """Structural interface ``simulate_policy`` drives any non-learned
    controller through: a ``lambda_price`` dict fed into the existing,
    unchanged ``price_utilities``, and an ``update`` hook called once per
    step with the raw ingredients every variant needs."""

    lambda_price: Dict[str, float]

    def update(self, context: PricingUpdateContext) -> None: ...


def total_available_over_horizon(seq: DynamicSequence) -> Dict[str, float]:
    """Initial budget plus every step's declared replenishment for the
    whole sequence -- treated as known service capacity (e.g. "this quota
    resets on a fixed schedule"), not a forecast of future demand or
    utility outcomes. See ADR 0004 for why this assumption is the least
    assumption-heavy way to make pacing meaningful."""
    total = dict(seq.initial_budget)
    for case in seq.cases:
        for r in seq.resource_names:
            total[r] += case.replenishment[r]
    return total


class ReactiveController:
    """Adapter reproducing tranche 3's exact reactive dual-pricing
    behavior against the new pluggable interface. Wraps ``DualController``
    unchanged -- no logic here duplicates or alters its update rule."""

    def __init__(
        self, resource_names: Tuple[str, ...], eta: float = 0.5, lambda_max: float = 20.0
    ) -> None:
        self._dual = DualController(resource_names=resource_names, eta=eta, lambda_max=lambda_max)

    @property
    def lambda_price(self) -> Dict[str, float]:
        return self._dual.lambda_price

    def update(self, context: PricingUpdateContext) -> None:
        steps_left = max(1, context.total_steps - context.step)
        utilization_error = {
            r: context.reservation.get(r, 0.0) - (context.remaining_after[r] / steps_left)
            for r in context.resource_names
        }
        self._dual.update(utilization_error)


@dataclass
class PacingController:
    """``lambda_{t+1,i} = clip(lambda_t,i + eta * h(e_t,i), 0, lambda_max)``
    where ``e_t,i`` compares the controller's own cumulative-reservation
    rate so far against the target rate implied by ``total_available``
    spread evenly across the horizon, and ``h`` is a deadband/asymmetric
    -relaxation function:

    ``h(e) = 0``                       if ``|e| <= deadband``
    ``h(e) = rise_scale * (e - deadband)``    if ``e > deadband``
    ``h(e) = relax_gamma * (e + deadband)``   if ``e < -deadband``

    Parameter choices realize four named variants (see the ADR): plain
    pacing (``deadband=0, relax_gamma=1``), pacing+hysteresis
    (``deadband>0``), asymmetric (``relax_gamma>1`` and/or
    ``rise_scale<1``), and bounded/smoothed (``ema_alpha`` and/or
    ``max_step`` set).
    """

    resource_names: Tuple[str, ...]
    total_available: Dict[str, float]
    total_steps: int
    eta: float = 0.5
    lambda_max: float = 20.0
    deadband: float = 0.0
    relax_gamma: float = 1.0
    rise_scale: float = 1.0
    ema_alpha: Optional[float] = None
    max_step: Optional[float] = None
    lambda_price: Dict[str, float] = field(default_factory=dict)
    _cumulative_usage: Dict[str, float] = field(default_factory=dict, repr=False)
    _smoothed_error: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.lambda_price:
            self.lambda_price = {r: 0.0 for r in self.resource_names}
        if not self._cumulative_usage:
            self._cumulative_usage = {r: 0.0 for r in self.resource_names}
        if not self._smoothed_error:
            self._smoothed_error = {r: 0.0 for r in self.resource_names}

    def _h(self, error: float) -> float:
        if abs(error) <= self.deadband:
            return 0.0
        if error > self.deadband:
            return self.rise_scale * (error - self.deadband)
        return self.relax_gamma * (error + self.deadband)

    def update(self, context: PricingUpdateContext) -> None:
        elapsed = max(context.step + 1, 1)
        for r in self.resource_names:
            self._cumulative_usage[r] += context.reservation.get(r, 0.0)
            actual_rate = self._cumulative_usage[r] / elapsed
            target_rate = self.total_available[r] / context.total_steps
            raw_error = actual_rate - target_rate
            if self.ema_alpha is not None:
                previous = self._smoothed_error[r]
                smoothed = (1.0 - self.ema_alpha) * previous + self.ema_alpha * raw_error
                self._smoothed_error[r] = smoothed
                error = smoothed
            else:
                error = raw_error
            delta_lambda = self.eta * self._h(error)
            if self.max_step is not None:
                delta_lambda = max(-self.max_step, min(self.max_step, delta_lambda))
            updated = self.lambda_price[r] + delta_lambda
            self.lambda_price[r] = min(max(updated, 0.0), self.lambda_max)
