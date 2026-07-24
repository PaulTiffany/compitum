"""Oracle-compatible pricing residual target -- exact hand-checked
intervals and the minimal-nudge residual computation."""

from __future__ import annotations

import pytest

from compitum.regret_lab.environment import DynamicCase
from compitum.regret_lab.residual_target import (
    compute_oracle_compatible_interval,
    oracle_price_residual,
)


def _case(base_utility, consumption):
    return DynamicCase(
        step=0,
        base_utility=base_utility,
        expected_consumption=consumption,
        realized_consumption=consumption,
        revelation_delay=0,
        replenishment={"budget": 0.0},
    )


def test_two_model_case_cheap_choice_needs_lambda_above_threshold() -> None:
    case = _case(
        {"cheap": 1.0, "expensive": 2.0},
        {"cheap": {"budget": 1.0}, "expensive": {"budget": 3.0}},
    )
    interval = compute_oracle_compatible_interval(case, "cheap")
    assert interval.feasible is True
    assert interval.low == pytest.approx(0.5)
    assert interval.high == pytest.approx(20.0)


def test_two_model_case_expensive_choice_needs_lambda_below_threshold() -> None:
    case = _case(
        {"cheap": 1.0, "expensive": 2.0},
        {"cheap": {"budget": 1.0}, "expensive": {"budget": 3.0}},
    )
    interval = compute_oracle_compatible_interval(case, "expensive")
    assert interval.feasible is True
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(0.5)


def test_three_model_case_gives_a_bounded_interval() -> None:
    case = _case(
        {"a": 1.0, "b": 2.0, "c": 3.0},
        {"a": {"budget": 1.0}, "b": {"budget": 2.0}, "c": {"budget": 4.0}},
    )
    # b beats a (2-2L > 1-L) when L < 1.0; b beats c (2-2L > 3-4L) when
    # L > 0.5 -- so b's compatible interval is (0.5, 1.0).
    interval = compute_oracle_compatible_interval(case, "b")
    assert interval.feasible is True
    assert interval.low == pytest.approx(0.5)
    assert interval.high == pytest.approx(1.0)


def test_strictly_dominated_choice_is_infeasible() -> None:
    # 'a' and 'b' cost the same but 'b' has strictly higher utility -> 'a'
    # can never win regardless of lambda.
    case = _case({"a": 1.0, "b": 3.0}, {"a": {"budget": 2.0}, "b": {"budget": 2.0}})
    interval = compute_oracle_compatible_interval(case, "a")
    assert interval.feasible is False


def test_defer_oracle_choice_is_always_infeasible() -> None:
    case = _case({"a": 1.0}, {"a": {"budget": 1.0}})
    interval = compute_oracle_compatible_interval(case, "defer")
    assert interval.feasible is False
    assert interval.low == 0.0


def test_low_exceeding_high_after_intersection_is_infeasible() -> None:
    # 'a' only beats 'b' for lambda <= -5 (never, for lambda>=0) and only
    # beats 'c' for lambda >= 0 -- the intersection [0, -5] is empty.
    case = _case(
        {"a": 1.0, "b": 1.0 + 5.0, "c": 1.0 - 1.0},
        {"a": {"budget": 0.0}, "b": {"budget": -1.0}, "c": {"budget": 1.0}},
    )
    interval = compute_oracle_compatible_interval(case, "a")
    assert interval.feasible is False


def test_same_consumption_but_oracle_has_higher_utility_adds_no_constraint() -> None:
    # 'a' (oracle) and 'b' cost the same, but 'a' has strictly higher
    # utility -- 'a' beats 'b' for every lambda, so this comparison must
    # not narrow the interval at all.
    case = _case(
        {"a": 3.0, "b": 1.0, "c": 2.0},
        {"a": {"budget": 2.0}, "b": {"budget": 2.0}, "c": {"budget": 5.0}},
    )
    interval = compute_oracle_compatible_interval(case, "a")
    assert interval.feasible is True
    # Only the 'c' comparison constrains: 3-2L > 2-5L -> L > -1/3, always
    # true for L>=0, so low stays 0 and high stays lambda_max.
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(20.0)


def test_oracle_price_residual_zero_when_lambda_base_inside_interval() -> None:
    case = _case(
        {"cheap": 1.0, "expensive": 2.0},
        {"cheap": {"budget": 1.0}, "expensive": {"budget": 3.0}},
    )
    interval = compute_oracle_compatible_interval(case, "cheap")
    assert oracle_price_residual(interval, lambda_base=5.0) == 0.0


def test_oracle_price_residual_positive_when_lambda_base_too_low() -> None:
    case = _case(
        {"cheap": 1.0, "expensive": 2.0},
        {"cheap": {"budget": 1.0}, "expensive": {"budget": 3.0}},
    )
    interval = compute_oracle_compatible_interval(case, "cheap")
    residual = oracle_price_residual(interval, lambda_base=0.1)
    assert residual == pytest.approx(0.4)  # 0.5 - 0.1


def test_oracle_price_residual_negative_when_lambda_base_too_high() -> None:
    case = _case(
        {"cheap": 1.0, "expensive": 2.0},
        {"cheap": {"budget": 1.0}, "expensive": {"budget": 3.0}},
    )
    interval = compute_oracle_compatible_interval(case, "expensive")
    residual = oracle_price_residual(interval, lambda_base=5.0)
    assert residual == pytest.approx(-4.5)  # 0.5 - 5.0


def test_oracle_price_residual_none_when_infeasible() -> None:
    case = _case({"a": 1.0, "b": 3.0}, {"a": {"budget": 2.0}, "b": {"budget": 2.0}})
    interval = compute_oracle_compatible_interval(case, "a")
    assert oracle_price_residual(interval, lambda_base=1.0) is None


def test_lambda_max_bounds_the_interval() -> None:
    case = _case(
        {"cheap": 1.0, "expensive": 2.0},
        {"cheap": {"budget": 1.0}, "expensive": {"budget": 3.0}},
    )
    interval = compute_oracle_compatible_interval(case, "cheap", lambda_max=0.1)
    # threshold 0.5 > lambda_max 0.1 -> low exceeds high -> infeasible
    assert interval.feasible is False
