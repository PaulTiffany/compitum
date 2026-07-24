"""Regret aggregation, paired deltas, and bootstrap CI -- with synthetic
PolicyRunResult/HindsightResult fixtures so the arithmetic is checked
directly, not just exercised."""

from __future__ import annotations

from compitum.regret_lab.hindsight import HindsightResult
from compitum.regret_lab.metrics import (
    PolicyRunResult,
    bootstrap_ci,
    paired_regret_deltas,
    regret_metrics,
)


def _policy_result(sequence_id, utility, **overrides):
    defaults = dict(
        sequence_id=sequence_id,
        cumulative_utility=utility,
        choices=["a", "b", "a"],
        violation_count=0,
        violation_magnitude=0.0,
        deferral_count=0,
        avoidable_deferral_count=0,
        route_switch_count=2,
        depleted_budget_events=0,
        total_consumption={"budget": 5.0, "quota": 2.0},
        decision_latencies=[0.001, 0.002],
    )
    defaults.update(overrides)
    return PolicyRunResult(**defaults)


def _hindsight(value):
    return HindsightResult(value=value, choices=[], exact=True, optimality_gap=0.0, state_count=1)


def test_regret_metrics_empty_input() -> None:
    metrics = regret_metrics([], {})
    assert metrics["n_sequences"] == 0.0
    assert metrics["mean_regret"] != metrics["mean_regret"]  # nan


def test_regret_metrics_basic_aggregation() -> None:
    results = [_policy_result("s1", 8.0), _policy_result("s2", 6.0)]
    hindsight = {"s1": _hindsight(10.0), "s2": _hindsight(10.0)}
    metrics = regret_metrics(results, hindsight)
    assert metrics["mean_regret"] == 3.0
    assert metrics["median_regret"] == 3.0
    assert metrics["n_sequences"] == 2.0


def test_regret_metrics_reports_violations_separately_from_regret() -> None:
    results = [
        _policy_result("s1", 10.0, violation_count=2, violation_magnitude=1.5),
        _policy_result("s2", 10.0),
    ]
    hindsight = {"s1": _hindsight(10.0), "s2": _hindsight(10.0)}
    metrics = regret_metrics(results, hindsight)
    assert metrics["mean_regret"] == 0.0  # no utility shortfall
    assert metrics["total_violation_count"] == 2.0  # but violations still visible
    assert metrics["total_violation_magnitude"] == 1.5


def test_route_switch_rate_normalizes_by_steps_minus_one() -> None:
    results = [_policy_result("s1", 5.0, choices=["a", "b", "a"], route_switch_count=2)]
    metrics = regret_metrics(results, {"s1": _hindsight(5.0)})
    assert metrics["mean_route_switch_rate"] == 1.0  # 2 switches / 2 transitions


def test_route_switch_rate_single_step_sequence_is_zero() -> None:
    results = [_policy_result("s1", 5.0, choices=["a"], route_switch_count=0)]
    metrics = regret_metrics(results, {"s1": _hindsight(5.0)})
    assert metrics["mean_route_switch_rate"] == 0.0


def test_utility_per_resource_unit() -> None:
    results = [_policy_result("s1", 10.0, total_consumption={"budget": 4.0, "quota": 1.0})]
    metrics = regret_metrics(results, {"s1": _hindsight(10.0)})
    assert metrics["utility_per_resource_unit"] == 2.0  # 10 / (4+1)


def test_utility_per_resource_unit_nan_when_no_consumption() -> None:
    results = [_policy_result("s1", 0.0, total_consumption={"budget": 0.0, "quota": 0.0})]
    metrics = regret_metrics(results, {"s1": _hindsight(0.0)})
    assert metrics["utility_per_resource_unit"] != metrics["utility_per_resource_unit"]  # nan


def test_paired_regret_deltas() -> None:
    a = [_policy_result("s1", 8.0)]
    b = [_policy_result("s1", 6.0)]
    hindsight = {"s1": _hindsight(10.0)}
    deltas = paired_regret_deltas(a, b, hindsight)
    # regret_a=2, regret_b=4 -> delta = -2 (A has less regret, i.e. is better)
    assert deltas == [-2.0]


def test_bootstrap_ci_empty() -> None:
    ci = bootstrap_ci([])
    assert ci["mean"] != ci["mean"]  # nan


def test_bootstrap_ci_all_negative_deltas_gives_negative_ci() -> None:
    deltas = [-1.0, -1.2, -0.9, -1.1] * 10
    ci = bootstrap_ci(deltas, n_resamples=200, seed=1)
    assert ci["ci_high"] < 0.0
    assert ci["mean"] < 0.0


def test_policy_run_result_to_dict_has_all_fields() -> None:
    result = _policy_result("s1", 5.0)
    d = result.to_dict()
    assert d["sequence_id"] == "s1"
    assert d["cumulative_utility"] == 5.0
    assert d["choices"] == ["a", "b", "a"]
    assert "total_consumption" in d
    assert "decision_latencies" in d
