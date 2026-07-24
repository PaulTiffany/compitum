"""Pluggable non-learned pricing controllers -- ReactiveController's exact
reproduction of tranche 3's formula, and PacingController's four
parameterized variants (plain, hysteresis, asymmetric, bounded/smoothed)."""

from __future__ import annotations

import pytest

from compitum.regret_lab.dual_controller import DualController
from compitum.regret_lab.environment import DynamicCase, DynamicSequence
from compitum.regret_lab.pricing import (
    PacingController,
    PricingUpdateContext,
    ReactiveController,
    total_available_over_horizon,
)


def _context(**overrides):
    defaults = dict(
        resource_names=("budget", "quota"),
        reservation={"budget": 1.0, "quota": 0.5},
        remaining_before={"budget": 5.0, "quota": 5.0},
        remaining_after={"budget": 4.0, "quota": 4.5},
        step=0,
        total_steps=10,
    )
    defaults.update(overrides)
    return PricingUpdateContext(**defaults)


def test_reactive_controller_matches_manual_dual_controller_update() -> None:
    reactive = ReactiveController(resource_names=("budget", "quota"), eta=0.5, lambda_max=20.0)
    manual = DualController(resource_names=("budget", "quota"), eta=0.5, lambda_max=20.0)

    context = _context()
    reactive.update(context)

    steps_left = max(1, context.total_steps - context.step)
    utilization_error = {
        r: context.reservation[r] - (context.remaining_after[r] / steps_left)
        for r in context.resource_names
    }
    manual.update(utilization_error)

    assert reactive.lambda_price == manual.lambda_price


def test_reactive_controller_lambda_price_property_reflects_wrapped_dual() -> None:
    reactive = ReactiveController(resource_names=("budget",), eta=1.0)
    reactive.update(
        _context(
            resource_names=("budget",), reservation={"budget": 5.0}, remaining_after={"budget": 1.0}
        )
    )
    assert reactive.lambda_price["budget"] > 0.0


def test_pacing_controller_raises_price_when_overusing() -> None:
    # total_available=10 over 10 steps -> target rate 1.0/step; reserving
    # 2.0 every step means actual rate (2.0) exceeds target -> price rises.
    controller = PacingController(
        resource_names=("budget",), total_available={"budget": 10.0}, total_steps=10, eta=1.0
    )
    controller.update(
        _context(resource_names=("budget",), reservation={"budget": 2.0}, step=0, total_steps=10)
    )
    assert controller.lambda_price["budget"] > 0.0


def test_pacing_controller_stays_at_zero_when_on_target() -> None:
    controller = PacingController(
        resource_names=("budget",), total_available={"budget": 10.0}, total_steps=10, eta=1.0
    )
    controller.update(
        _context(resource_names=("budget",), reservation={"budget": 1.0}, step=0, total_steps=10)
    )
    assert controller.lambda_price["budget"] == pytest.approx(0.0)


def test_pacing_controller_lambda_never_negative_or_above_max() -> None:
    controller = PacingController(
        resource_names=("budget",),
        total_available={"budget": 100.0},
        total_steps=10,
        eta=1.0,
        lambda_max=5.0,
    )
    for step in range(10):
        controller.update(
            _context(
                resource_names=("budget",), reservation={"budget": 50.0}, step=step, total_steps=10
            )
        )
    assert controller.lambda_price["budget"] == 5.0


def test_hysteresis_deadband_suppresses_small_errors() -> None:
    # Same mild over-use as the "raises price" test, but with a deadband
    # wide enough to absorb it -- price must stay at zero.
    controller = PacingController(
        resource_names=("budget",),
        total_available={"budget": 10.0},
        total_steps=10,
        eta=1.0,
        deadband=2.0,
    )
    controller.update(
        _context(resource_names=("budget",), reservation={"budget": 2.0}, step=0, total_steps=10)
    )
    assert controller.lambda_price["budget"] == 0.0


def test_hysteresis_deadband_still_responds_to_large_errors() -> None:
    controller = PacingController(
        resource_names=("budget",),
        total_available={"budget": 10.0},
        total_steps=10,
        eta=1.0,
        deadband=0.1,
    )
    controller.update(
        _context(resource_names=("budget",), reservation={"budget": 5.0}, step=0, total_steps=10)
    )
    assert controller.lambda_price["budget"] > 0.0


def test_asymmetric_relaxes_faster_than_it_rises() -> None:
    rise = PacingController(
        resource_names=("budget",),
        total_available={"budget": 10.0},
        total_steps=10,
        eta=1.0,
        relax_gamma=3.0,
    )
    # Overuse step (rises using rise_scale=1, no relax_gamma involved).
    rise.update(
        _context(resource_names=("budget",), reservation={"budget": 3.0}, step=0, total_steps=10)
    )
    risen = rise.lambda_price["budget"]
    assert risen > 0.0

    relax = PacingController(
        resource_names=("budget",),
        total_available={"budget": 10.0},
        total_steps=10,
        eta=1.0,
        relax_gamma=3.0,
    )
    relax.lambda_price["budget"] = risen
    relax._cumulative_usage["budget"] = 0.0
    # Heavy under-use at step 1 -> negative error -> relax_gamma amplifies
    # the relaxation compared to a symmetric (relax_gamma=1) controller.
    relax.update(
        _context(resource_names=("budget",), reservation={"budget": 0.0}, step=1, total_steps=10)
    )

    symmetric = PacingController(
        resource_names=("budget",), total_available={"budget": 10.0}, total_steps=10, eta=1.0
    )
    symmetric.lambda_price["budget"] = risen
    symmetric._cumulative_usage["budget"] = 0.0
    symmetric.update(
        _context(resource_names=("budget",), reservation={"budget": 0.0}, step=1, total_steps=10)
    )

    assert relax.lambda_price["budget"] < symmetric.lambda_price["budget"]


def test_ema_smoothing_dampens_a_single_step_spike() -> None:
    smoothed = PacingController(
        resource_names=("budget",),
        total_available={"budget": 10.0},
        total_steps=10,
        eta=1.0,
        ema_alpha=0.1,
    )
    unsmoothed = PacingController(
        resource_names=("budget",), total_available={"budget": 10.0}, total_steps=10, eta=1.0
    )
    context = _context(
        resource_names=("budget",), reservation={"budget": 10.0}, step=0, total_steps=10
    )
    smoothed.update(context)
    unsmoothed.update(context)
    assert smoothed.lambda_price["budget"] < unsmoothed.lambda_price["budget"]


def test_max_step_bounds_the_per_step_lambda_change() -> None:
    controller = PacingController(
        resource_names=("budget",),
        total_available={"budget": 10.0},
        total_steps=10,
        eta=1.0,
        max_step=0.05,
    )
    controller.update(
        _context(resource_names=("budget",), reservation={"budget": 10.0}, step=0, total_steps=10)
    )
    assert controller.lambda_price["budget"] == pytest.approx(0.05)


def test_pacing_controller_explicit_lambda_price_constructor_arg() -> None:
    controller = PacingController(
        resource_names=("budget",),
        total_available={"budget": 10.0},
        total_steps=10,
        lambda_price={"budget": 3.0},
    )
    assert controller.lambda_price == {"budget": 3.0}


def _seq(steps: int, initial_budget, replenishment):
    cases = [
        DynamicCase(
            step=t,
            base_utility={"a": 1.0},
            expected_consumption={"a": {"budget": 1.0}},
            realized_consumption={"a": {"budget": 1.0}},
            revelation_delay=0,
            replenishment=replenishment,
        )
        for t in range(steps)
    ]
    return DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget",),
        model_names=("a",),
        initial_budget=initial_budget,
        cases=cases,
    )


def test_pacing_controller_explicit_internal_state_constructor_args() -> None:
    controller = PacingController(
        resource_names=("budget",),
        total_available={"budget": 10.0},
        total_steps=10,
        _cumulative_usage={"budget": 2.0},
        _smoothed_error={"budget": 0.5},
    )
    assert controller._cumulative_usage == {"budget": 2.0}
    assert controller._smoothed_error == {"budget": 0.5}


def test_total_available_over_horizon_sums_initial_and_all_replenishment() -> None:
    seq = _seq(4, {"budget": 10.0}, {"budget": 2.0})
    total = total_available_over_horizon(seq)
    assert total == {"budget": 10.0 + 4 * 2.0}
