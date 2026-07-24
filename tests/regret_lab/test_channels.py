"""Declared FabricPC channel mapping for the dynamic-regret environment --
fixed order, dependency-free, verified field-by-field."""

from __future__ import annotations

from compitum.regret_lab.channels import CHANNEL_DIMENSION, compute_regret_channel_vector
from compitum.regret_lab.environment import MODEL_NAMES, RESOURCE_NAMES, DynamicCase


def _case(**overrides):
    defaults = dict(
        step=0,
        base_utility={"economy": 1.0, "standard": 2.0, "premium": 3.5},
        expected_consumption={
            "economy": {"budget": 0.25, "quota": 0.25},
            "standard": {"budget": 0.75, "quota": 0.5},
            "premium": {"budget": 1.5, "quota": 1.25},
        },
        realized_consumption={
            "economy": {"budget": 0.25, "quota": 0.25},
            "standard": {"budget": 0.75, "quota": 0.5},
            "premium": {"budget": 1.5, "quota": 1.25},
        },
        revelation_delay=0,
        replenishment={"budget": 0.5, "quota": 0.5},
    )
    defaults.update(overrides)
    return DynamicCase(**defaults)


def test_vector_has_declared_dimension() -> None:
    vector = compute_regret_channel_vector(
        {"budget": 5.0, "quota": 5.0}, _case(), {"budget": 0.0, "quota": 0.0}, 5, 10
    )
    assert vector.shape == (CHANNEL_DIMENSION,)


def test_remaining_budget_is_normalized() -> None:
    vector = compute_regret_channel_vector(
        {"budget": 10.0, "quota": 5.0}, _case(), {"budget": 0.0, "quota": 0.0}, 5, 10
    )
    assert vector[0] == 1.0  # 10 / _BUDGET_NORM(10.0)
    assert vector[1] == 0.5


def test_expected_consumption_block_matches_model_resource_order() -> None:
    vector = compute_regret_channel_vector(
        {"budget": 5.0, "quota": 5.0}, _case(), {"budget": 0.0, "quota": 0.0}, 5, 10
    )
    idx = 2
    for m in MODEL_NAMES:
        for r in RESOURCE_NAMES:
            assert vector[idx] == _case().expected_consumption[m][r]
            idx += 1


def test_base_utility_block() -> None:
    vector = compute_regret_channel_vector(
        {"budget": 5.0, "quota": 5.0}, _case(), {"budget": 0.0, "quota": 0.0}, 5, 10
    )
    assert list(vector[8:11]) == [1.0, 2.0, 3.5]


def test_lambda_price_block() -> None:
    vector = compute_regret_channel_vector(
        {"budget": 5.0, "quota": 5.0}, _case(), {"budget": 1.5, "quota": 2.5}, 5, 10
    )
    assert vector[11] == 1.5
    assert vector[12] == 2.5


def test_steps_left_fraction() -> None:
    vector = compute_regret_channel_vector(
        {"budget": 5.0, "quota": 5.0}, _case(), {"budget": 0.0, "quota": 0.0}, 5, 10
    )
    assert vector[13] == 0.5


def test_steps_left_fraction_zero_total_steps_is_zero() -> None:
    vector = compute_regret_channel_vector(
        {"budget": 5.0, "quota": 5.0}, _case(), {"budget": 0.0, "quota": 0.0}, 0, 0
    )
    assert vector[13] == 0.0


def test_replenishment_total() -> None:
    vector = compute_regret_channel_vector(
        {"budget": 5.0, "quota": 5.0},
        _case(replenishment={"budget": 0.5, "quota": 0.25}),
        {"budget": 0.0, "quota": 0.0},
        5,
        10,
    )
    assert vector[14] == 0.75


def test_missing_resource_in_remaining_defaults_to_zero() -> None:
    vector = compute_regret_channel_vector({}, _case(), {"budget": 0.0, "quota": 0.0}, 5, 10)
    assert vector[0] == 0.0
    assert vector[1] == 0.0
