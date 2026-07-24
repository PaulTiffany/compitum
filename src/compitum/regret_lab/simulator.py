"""Online policy simulator (tranche 3).

Runs one policy across a full ``DynamicSequence``, applying a reservation
-then-true-up ledger (immediate reservation using the forecast available at
decision time; a correction applied once ``revelation_delay`` elapses,
which can push the ledger negative -- recorded as a genuine violation, not
hidden). Never reads or writes production Compitum state; this is a
self-contained offline simulation, per docs/adr/0003-dynamic-constraint-regret.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dual_controller import price_utilities
from .environment import DynamicCase, DynamicSequence
from .metrics import PolicyRunResult
from .pricing import PricingController, PricingUpdateContext


@dataclass
class ForecastContext:
    """Extra decision-time state a forecaster may use beyond the raw
    ``expected_consumption`` it is handed -- e.g. a FabricPC-backed
    forecaster needs the live remaining budget and dual price to build its
    declared observation channel (``regret_lab.channels``). Simple
    forecasters (like ``EWMAForecaster``) ignore this entirely."""

    case: DynamicCase
    remaining: Dict[str, float]
    lambda_price: Dict[str, float]
    steps_left: int
    total_steps: int


Forecaster = Callable[[Dict[str, Dict[str, float]], ForecastContext], Dict[str, Dict[str, float]]]


@dataclass
class PolicyDecision:
    step: int
    chosen: str
    feasible_models: List[str]
    priced_utility: Dict[str, float]
    violation_magnitude_so_far: float
    latency_seconds: float
    remaining_before: Dict[str, float]
    lambda_price_before: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "chosen": self.chosen,
            "feasible_models": list(self.feasible_models),
            "priced_utility": dict(self.priced_utility),
            "violation_magnitude_so_far": self.violation_magnitude_so_far,
            "latency_seconds": self.latency_seconds,
            "remaining_before": dict(self.remaining_before),
            "lambda_price_before": dict(self.lambda_price_before),
        }


@dataclass
class _PendingCorrection:
    due_step: int
    resource_deltas: Dict[str, float]


def simulate_policy(
    seq: DynamicSequence,
    pricing_controller: Optional[PricingController] = None,
    forecaster: Optional[Forecaster] = None,
    forecaster_update: Optional[Callable[[str, Dict[str, float], Dict[str, float]], None]] = None,
) -> Tuple[PolicyRunResult, List[PolicyDecision]]:
    """``pricing_controller=None`` reproduces the static/frozen arm (greedy
    on raw ``base_utility`` among expected-feasible models, no pricing).
    Any object exposing ``lambda_price``/``update(context)`` (see
    ``regret_lab.pricing``) may be passed -- e.g. ``ReactiveController``
    (reproduces tranche 3's reactive dual controller) or
    ``PacingController`` (tranche 4's non-learned pacing family).
    ``forecaster`` (if given) replaces ``case.expected_consumption`` with a
    prediction for both the feasibility check and pricing; pass
    ``forecaster_update`` (e.g. an ``EWMAForecaster.update``-bound method)
    to let it learn from the chosen model's realized outcome each step."""
    remaining = dict(seq.initial_budget)
    pending: List[_PendingCorrection] = []
    cumulative_utility = 0.0
    choices: List[str] = []
    decisions: List[PolicyDecision] = []
    violation_count = 0
    violation_magnitude_total = 0.0
    deferral_count = 0
    avoidable_deferral_count = 0
    depleted_budget_events = 0
    high_value_rejection_count = 0
    total_consumption: Dict[str, float] = {r: 0.0 for r in seq.resource_names}

    def _apply_delta(resource: str, delta: float) -> None:
        nonlocal violation_count, violation_magnitude_total
        remaining[resource] += delta
        if remaining[resource] < 0:
            violation_magnitude_total += -remaining[resource]
            violation_count += 1
            remaining[resource] = 0.0

    for t, case in enumerate(seq.cases):
        start = time.perf_counter()

        still_pending = []
        for corr in pending:
            if corr.due_step <= t:
                for r, correction in corr.resource_deltas.items():
                    _apply_delta(r, correction)
            else:
                still_pending.append(corr)
        pending = still_pending

        remaining_before = dict(remaining)
        lambda_price_before = dict(pricing_controller.lambda_price) if pricing_controller else {}

        if forecaster is not None:
            context = ForecastContext(
                case=case,
                remaining=remaining_before,
                lambda_price=lambda_price_before,
                steps_left=len(seq.cases) - t,
                total_steps=len(seq.cases),
            )
            forecast = forecaster(case.expected_consumption, context)
        else:
            forecast = case.expected_consumption
        feasible = [
            m
            for m in seq.model_names
            if all(remaining[r] - forecast[m][r] >= -1e-9 for r in seq.resource_names)
        ]

        if pricing_controller is not None:
            priced = price_utilities(case.base_utility, forecast, pricing_controller.lambda_price)
        else:
            priced = dict(case.base_utility)

        if feasible:
            chosen = max(feasible, key=lambda m: priced[m])
            best_by_base_utility = max(feasible, key=lambda m: case.base_utility[m])
            # A direct, per-step signature of hoarding: pricing picked a
            # lower-utility model even though a higher-utility one was
            # genuinely affordable (checked against REALIZED consumption,
            # not just the forecast the pricing decision itself used).
            if chosen != best_by_base_utility and all(
                remaining[r] - case.realized_consumption[best_by_base_utility][r] >= -1e-9
                for r in seq.resource_names
            ):
                high_value_rejection_count += 1
        else:
            chosen = "defer"
            deferral_count += 1
            # "Unnecessary"/avoidable: the forecast blocked every model, but
            # at least one model's true realized consumption would actually
            # have fit -- a real capacity existed, only the forecast hid it.
            if any(
                all(
                    remaining[r] - case.realized_consumption[m][r] >= -1e-9
                    for r in seq.resource_names
                )
                for m in seq.model_names
            ):
                avoidable_deferral_count += 1

        if chosen != "defer":
            cumulative_utility += case.base_utility[chosen]
            reservation = forecast[chosen]
            for r in seq.resource_names:
                _apply_delta(r, -reservation[r])
            realized = case.realized_consumption[chosen]
            delta = {r: reservation[r] - realized[r] for r in seq.resource_names}
            if case.revelation_delay <= 0:
                for r in seq.resource_names:
                    _apply_delta(r, delta[r])
            else:
                pending.append(
                    _PendingCorrection(due_step=t + case.revelation_delay, resource_deltas=delta)
                )
            if forecaster_update is not None:
                forecaster_update(chosen, case.expected_consumption[chosen], realized)
            for r in seq.resource_names:
                total_consumption[r] += realized[r]

        for r in seq.resource_names:
            remaining[r] += case.replenishment[r]

        if any(remaining[r] <= 1e-9 for r in seq.resource_names):
            depleted_budget_events += 1

        if pricing_controller is not None and chosen != "defer":
            update_context = PricingUpdateContext(
                resource_names=seq.resource_names,
                reservation=forecast[chosen],
                remaining_before=remaining_before,
                remaining_after=dict(remaining),
                step=t,
                total_steps=len(seq.cases),
                case=case,
                chosen=chosen,
            )
            pricing_controller.update(update_context)

        latency = time.perf_counter() - start
        decisions.append(
            PolicyDecision(
                step=t,
                chosen=chosen,
                feasible_models=feasible,
                priced_utility=priced,
                violation_magnitude_so_far=violation_magnitude_total,
                latency_seconds=latency,
                remaining_before=remaining_before,
                lambda_price_before=lambda_price_before,
            )
        )
        choices.append(chosen)

    route_switch_count = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i - 1])

    result = PolicyRunResult(
        sequence_id=seq.sequence_id,
        cumulative_utility=cumulative_utility,
        choices=choices,
        violation_count=violation_count,
        violation_magnitude=violation_magnitude_total,
        deferral_count=deferral_count,
        avoidable_deferral_count=avoidable_deferral_count,
        route_switch_count=route_switch_count,
        depleted_budget_events=depleted_budget_events,
        total_consumption=total_consumption,
        decision_latencies=[d.latency_seconds for d in decisions],
        terminal_remaining=dict(remaining),
        high_value_rejections=high_value_rejection_count,
    )
    return result, decisions
