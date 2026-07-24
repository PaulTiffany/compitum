"""Bounded, gated residual-price adapter -- clipping, governed failure
fallback, gating, and window bookkeeping."""

from __future__ import annotations

import pytest

from compitum.regret_lab.environment import DynamicCase
from compitum.regret_lab.pricing import PacingController, PricingUpdateContext
from compitum.regret_lab.residual_pricing import ResidualPricingController


def _case(**overrides):
    defaults = dict(
        step=0,
        base_utility={"conserve": 1.0, "spend": 2.0, "opportunity": 0.0},
        expected_consumption={
            "conserve": {"budget": 1.0},
            "spend": {"budget": 2.0},
            "opportunity": {"budget": 1.0e6},
        },
        realized_consumption={
            "conserve": {"budget": 1.0},
            "spend": {"budget": 2.0},
            "opportunity": {"budget": 1.0e6},
        },
        revelation_delay=0,
        replenishment={"budget": 0.5},
    )
    defaults.update(overrides)
    return DynamicCase(**defaults)


def _context(step=0, total_steps=10, chosen="spend", case=None):
    return PricingUpdateContext(
        resource_names=("budget",),
        reservation={"budget": 2.0},
        remaining_before={"budget": 10.0},
        remaining_after={"budget": 8.0},
        step=step,
        total_steps=total_steps,
        case=case or _case(),
        chosen=chosen,
    )


def _base_controller():
    return PacingController(
        resource_names=("budget",), total_available={"budget": 20.0}, total_steps=10, eta=1.0
    )


def test_correction_applied_within_bounds() -> None:
    controller = ResidualPricingController(
        base=_base_controller(),
        predict_residual=lambda window: 0.5,
        max_correction_magnitude=2.0,
    )
    controller.update(_context())
    assert controller.records[-1].status == "applied"
    assert controller.records[-1].applied_correction == pytest.approx(0.5)
    assert controller.lambda_price["budget"] == pytest.approx(
        controller.base.lambda_price["budget"] + 0.5
    )


def test_correction_clipped_to_max_magnitude() -> None:
    controller = ResidualPricingController(
        base=_base_controller(),
        predict_residual=lambda window: 100.0,
        max_correction_magnitude=1.5,
    )
    controller.update(_context())
    assert controller.records[-1].status == "clipped"
    assert controller.records[-1].applied_correction == pytest.approx(1.5)


def test_negative_correction_clipped_symmetrically() -> None:
    controller = ResidualPricingController(
        base=_base_controller(),
        predict_residual=lambda window: -100.0,
        max_correction_magnitude=1.5,
    )
    controller.update(_context())
    assert controller.records[-1].applied_correction == pytest.approx(-1.5)


def test_predictor_exception_falls_back_to_zero() -> None:
    def _boom(window):
        raise RuntimeError("observer down")

    controller = ResidualPricingController(
        base=_base_controller(), predict_residual=_boom, max_correction_magnitude=1.0
    )
    controller.update(_context())
    assert controller.records[-1].status == "failed"
    assert controller.records[-1].applied_correction == 0.0
    assert controller.lambda_price["budget"] == controller.base.lambda_price["budget"]


def test_predictor_returning_none_falls_back_to_zero() -> None:
    controller = ResidualPricingController(
        base=_base_controller(), predict_residual=lambda window: None, max_correction_magnitude=1.0
    )
    controller.update(_context())
    assert controller.records[-1].status == "failed"
    assert controller.records[-1].applied_correction == 0.0


def test_gate_closed_forces_zero_correction_regardless_of_predictor() -> None:
    controller = ResidualPricingController(
        base=_base_controller(),
        predict_residual=lambda window: 5.0,
        max_correction_magnitude=10.0,
        gate_fn=lambda context, lambda_base: False,
    )
    controller.update(_context())
    assert controller.records[-1].status == "zero_gate"
    assert controller.records[-1].applied_correction == 0.0


def test_gate_fn_receives_the_current_base_lambda() -> None:
    seen_lambda_bases = []

    def _gate(context, lambda_base):
        seen_lambda_bases.append(lambda_base)
        return True

    controller = ResidualPricingController(
        base=_base_controller(),
        predict_residual=lambda window: 0.0,
        max_correction_magnitude=1.0,
        gate_fn=_gate,
    )
    controller.update(_context())
    assert seen_lambda_bases == [controller.base.lambda_price["budget"]]


def test_missing_case_or_chosen_raises() -> None:
    controller = ResidualPricingController(
        base=_base_controller(), predict_residual=lambda window: 0.0, max_correction_magnitude=1.0
    )
    bad_context = PricingUpdateContext(
        resource_names=("budget",),
        reservation={"budget": 1.0},
        remaining_before={"budget": 5.0},
        remaining_after={"budget": 4.0},
        step=0,
        total_steps=10,
    )
    with pytest.raises(ValueError, match="requires"):
        controller.update(bad_context)


def test_window_never_exceeds_declared_size() -> None:
    seen_window_sizes = []

    def _record_window(window):
        seen_window_sizes.append(len(window))
        return 0.0

    controller = ResidualPricingController(
        base=_base_controller(),
        predict_residual=_record_window,
        max_correction_magnitude=1.0,
        window_size=3,
    )
    for step in range(10):
        controller.update(_context(step=step))
    assert seen_window_sizes[-1] == 3
    assert max(seen_window_sizes) == 3


def test_records_accumulate_one_per_update() -> None:
    controller = ResidualPricingController(
        base=_base_controller(), predict_residual=lambda window: 0.0, max_correction_magnitude=1.0
    )
    for step in range(4):
        controller.update(_context(step=step))
    assert len(controller.records) == 4
    assert [r.step for r in controller.records] == [0, 1, 2, 3]


def test_lambda_price_never_goes_negative() -> None:
    controller = ResidualPricingController(
        base=_base_controller(),
        predict_residual=lambda window: -100.0,
        max_correction_magnitude=100.0,
    )
    controller.update(_context())
    assert controller.lambda_price["budget"] == 0.0


def test_lambda_price_respects_lambda_max() -> None:
    controller = ResidualPricingController(
        base=_base_controller(),
        predict_residual=lambda window: 100.0,
        max_correction_magnitude=100.0,
        lambda_max=5.0,
    )
    controller.update(_context())
    assert controller.lambda_price["budget"] == 5.0


def test_record_to_dict_has_all_fields() -> None:
    controller = ResidualPricingController(
        base=_base_controller(), predict_residual=lambda window: 0.25, max_correction_magnitude=1.0
    )
    controller.update(_context())
    d = controller.records[-1].to_dict()
    assert set(d) == {"step", "status", "raw_correction", "applied_correction", "window_size"}
