"""Declared belief-estimation channel -- fixed order, dependency-free,
verified field-by-field."""

from __future__ import annotations

from compitum.regret_lab.belief_channels import (
    CHANNEL_DIMENSION,
    BeliefChannelHistory,
    advance_belief_history,
    compute_belief_channel_vector,
)
from compitum.regret_lab.environment import DynamicCase


def _case(opportunity_available: bool = False):
    return DynamicCase(
        step=0,
        base_utility={
            "conserve": 1.0,
            "spend": 2.0,
            "opportunity": 8.0 if opportunity_available else 0.0,
        },
        expected_consumption={
            "conserve": {"budget": 1.0},
            "spend": {"budget": 2.0},
            "opportunity": {"budget": 4.0 if opportunity_available else 1.0e6},
        },
        realized_consumption={
            "conserve": {"budget": 1.0},
            "spend": {"budget": 2.0},
            "opportunity": {"budget": 4.0 if opportunity_available else 1.0e6},
        },
        revelation_delay=0,
        replenishment={"budget": 0.5},
    )


def test_vector_has_declared_dimension() -> None:
    vector = compute_belief_channel_vector(5.0, _case(), BeliefChannelHistory(), 5)
    assert vector.shape == (CHANNEL_DIMENSION,)


def test_no_previous_route_leaves_one_hot_block_zero() -> None:
    vector = compute_belief_channel_vector(5.0, _case(), BeliefChannelHistory(), 5)
    assert list(vector[0:4]) == [0.0, 0.0, 0.0, 0.0]


def test_previous_route_sets_correct_one_hot_slot() -> None:
    for route, index in (("conserve", 0), ("spend", 1), ("opportunity", 2), ("defer", 3)):
        history = BeliefChannelHistory(previous_route=route)
        vector = compute_belief_channel_vector(5.0, _case(), history, 5)
        expected = [0.0, 0.0, 0.0, 0.0]
        expected[index] = 1.0
        assert list(vector[0:4]) == expected


def test_previous_step_scalars() -> None:
    history = BeliefChannelHistory(
        previous_realized_consumption=2.0, previous_realized_utility=1.5, previous_replenishment=0.5
    )
    vector = compute_belief_channel_vector(5.0, _case(), history, 5)
    assert vector[4] == 2.0
    assert vector[5] == 1.5
    assert vector[6] == 0.5


def test_remaining_budget_normalized() -> None:
    vector = compute_belief_channel_vector(10.0, _case(), BeliefChannelHistory(), 5)
    assert vector[7] == 0.5  # 10 / BUDGET_NORM(20)


def test_steps_left_fraction() -> None:
    vector = compute_belief_channel_vector(5.0, _case(), BeliefChannelHistory(), 5, total_steps=10)
    assert vector[8] == 0.5


def test_steps_left_fraction_zero_total_steps() -> None:
    vector = compute_belief_channel_vector(5.0, _case(), BeliefChannelHistory(), 0, total_steps=0)
    assert vector[8] == 0.0


def test_opportunity_now_indicator() -> None:
    v_no_opp = compute_belief_channel_vector(5.0, _case(False), BeliefChannelHistory(), 5)
    v_with_opp = compute_belief_channel_vector(5.0, _case(True), BeliefChannelHistory(), 5)
    assert v_no_opp[9] == 0.0
    assert v_with_opp[9] == 1.0


def test_recent_opportunity_frequency_empty_history_is_zero() -> None:
    vector = compute_belief_channel_vector(5.0, _case(), BeliefChannelHistory(), 5)
    assert vector[10] == 0.0


def test_recent_opportunity_frequency_reflects_recent_history() -> None:
    history = BeliefChannelHistory()
    history.recent_opportunities.extend([True, True, False, False])
    vector = compute_belief_channel_vector(5.0, _case(), history, 5)
    assert vector[10] == 0.5


def test_advance_belief_history_records_chosen_route_outcome() -> None:
    history = BeliefChannelHistory()
    updated = advance_belief_history(history, "spend", _case(False))
    assert updated.previous_route == "spend"
    assert updated.previous_realized_consumption == 2.0
    assert updated.previous_realized_utility == 2.0
    assert updated.previous_replenishment == 0.5


def test_advance_belief_history_defer_records_zero_outcome() -> None:
    history = BeliefChannelHistory()
    updated = advance_belief_history(history, "defer", _case(False))
    assert updated.previous_route == "defer"
    assert updated.previous_realized_consumption == 0.0
    assert updated.previous_realized_utility == 0.0


def test_advance_belief_history_tracks_opportunity_recency() -> None:
    history = BeliefChannelHistory()
    history = advance_belief_history(history, "spend", _case(True))
    assert list(history.recent_opportunities) == [True]
    history = advance_belief_history(history, "conserve", _case(False))
    assert list(history.recent_opportunities) == [True, False]


def test_recent_opportunities_window_is_bounded() -> None:
    history = BeliefChannelHistory()
    for _ in range(10):
        history = advance_belief_history(history, "conserve", _case(False))
    assert len(history.recent_opportunities) == 5  # RECENT_OPPORTUNITY_WINDOW
