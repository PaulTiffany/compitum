"""Declared residual-pricing channel -- fixed order, dependency-free,
verified field-by-field."""

from __future__ import annotations

from compitum.regret_lab.environment import DynamicCase
from compitum.regret_lab.residual_channels import (
    CHANNEL_DIMENSION,
    OPPORTUNITY_SEEN_CAP,
    ResidualChannelHistory,
    advance_history,
    compute_residual_channel_vector,
)


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


def test_vector_has_declared_dimension() -> None:
    vector = compute_residual_channel_vector(
        remaining=10.0,
        case=_case(),
        lambda_base=1.0,
        pacing_error=0.5,
        history=ResidualChannelHistory(),
        steps_left=5,
        total_steps=10,
    )
    assert vector.shape == (CHANNEL_DIMENSION,)


def test_remaining_and_pacing_error_and_replenishment() -> None:
    vector = compute_residual_channel_vector(
        remaining=10.0,
        case=_case(),
        lambda_base=1.0,
        pacing_error=0.5,
        history=ResidualChannelHistory(),
        steps_left=5,
        total_steps=10,
    )
    assert vector[0] == 0.5  # 10 / BUDGET_NORM(20)
    assert vector[1] == 0.5
    assert vector[2] == 0.5


def test_expected_consumption_block_and_opportunity_clip() -> None:
    vector = compute_residual_channel_vector(
        remaining=10.0,
        case=_case(),
        lambda_base=0.0,
        pacing_error=0.0,
        history=ResidualChannelHistory(),
        steps_left=5,
        total_steps=10,
    )
    assert vector[3] == 1.0
    assert vector[4] == 2.0
    assert vector[5] == 50.0  # clipped from 1e6


def test_utility_gap_uses_priced_utility_at_lambda_base() -> None:
    # priced: conserve=1-1*1=0; spend=2-1*2=0; opportunity huge negative.
    vector = compute_residual_channel_vector(
        remaining=10.0,
        case=_case(),
        lambda_base=1.0,
        pacing_error=0.0,
        history=ResidualChannelHistory(),
        steps_left=5,
        total_steps=10,
    )
    assert vector[6] == 0.0  # conserve and spend tie at lambda=1


def test_lambda_and_lambda_change() -> None:
    history = ResidualChannelHistory(previous_lambda=0.5)
    vector = compute_residual_channel_vector(
        remaining=10.0,
        case=_case(),
        lambda_base=1.5,
        pacing_error=0.0,
        history=history,
        steps_left=5,
        total_steps=10,
    )
    assert vector[7] == 1.5
    assert vector[8] == 1.0  # 1.5 - 0.5


def test_route_switch_indicator_requires_two_prior_routes() -> None:
    history_no_switch = ResidualChannelHistory(
        previous_route="spend", route_before_previous="spend"
    )
    history_switch = ResidualChannelHistory(
        previous_route="spend", route_before_previous="conserve"
    )
    history_missing = ResidualChannelHistory(previous_route="spend", route_before_previous=None)

    v_no_switch = compute_residual_channel_vector(10.0, _case(), 0.0, 0.0, history_no_switch, 5, 10)
    v_switch = compute_residual_channel_vector(10.0, _case(), 0.0, 0.0, history_switch, 5, 10)
    v_missing = compute_residual_channel_vector(10.0, _case(), 0.0, 0.0, history_missing, 5, 10)
    assert v_no_switch[9] == 0.0
    assert v_switch[9] == 1.0
    assert v_missing[9] == 0.0


def test_steps_left_fraction() -> None:
    vector = compute_residual_channel_vector(
        10.0, _case(), 0.0, 0.0, ResidualChannelHistory(), 5, 10
    )
    assert vector[10] == 0.5


def test_steps_left_fraction_zero_total_steps() -> None:
    vector = compute_residual_channel_vector(
        10.0, _case(), 0.0, 0.0, ResidualChannelHistory(), 0, 0
    )
    assert vector[10] == 0.0


def test_opportunity_seen_recency_and_forecast_error() -> None:
    history = ResidualChannelHistory(steps_since_opportunity_seen=4, last_forecast_error=0.3)
    vector = compute_residual_channel_vector(10.0, _case(), 0.0, 0.0, history, 5, 10)
    assert vector[11] == 4 / OPPORTUNITY_SEEN_CAP
    assert vector[12] == 0.3


def test_advance_history_resets_recency_when_opportunity_seen_now() -> None:
    history = ResidualChannelHistory(steps_since_opportunity_seen=10)
    case_with_opportunity = _case(base_utility={"conserve": 1.0, "spend": 2.0, "opportunity": 6.0})
    updated = advance_history(history, case_with_opportunity, chosen="spend", lambda_base=1.0)
    assert updated.steps_since_opportunity_seen == 0
    assert updated.previous_route == "spend"
    assert updated.previous_lambda == 1.0


def test_advance_history_increments_recency_when_no_opportunity_and_caps() -> None:
    history = ResidualChannelHistory(steps_since_opportunity_seen=OPPORTUNITY_SEEN_CAP)
    updated = advance_history(history, _case(), chosen="conserve", lambda_base=0.5)
    assert updated.steps_since_opportunity_seen == OPPORTUNITY_SEEN_CAP  # capped, not exceeded
    assert updated.route_before_previous is None  # carried from history.previous_route


def test_advance_history_shifts_route_history() -> None:
    history = ResidualChannelHistory(previous_route="conserve", route_before_previous="spend")
    updated = advance_history(history, _case(), chosen="spend", lambda_base=0.0)
    assert updated.previous_route == "spend"
    assert updated.route_before_previous == "conserve"
