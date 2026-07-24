"""Conservation/depletion regret-attribution diagnostic -- a per-step
heuristic, verified against hand-constructed cases with known splits."""

from __future__ import annotations

import pytest

from compitum.regret_lab.diagnostics import conservation_depletion_split
from compitum.regret_lab.environment import DynamicCase, DynamicSequence
from compitum.regret_lab.hindsight import HindsightResult
from compitum.regret_lab.simulator import PolicyDecision


def _case(step, base_utility, consumption=None, replenishment=None):
    consumption = consumption or {m: {"budget": 0.0} for m in base_utility}
    return DynamicCase(
        step=step,
        base_utility=base_utility,
        expected_consumption=consumption,
        realized_consumption=consumption,
        revelation_delay=0,
        replenishment=replenishment or {"budget": 0.0},
    )


def _decision(chosen, remaining_before):
    return PolicyDecision(
        step=0,
        chosen=chosen,
        feasible_models=[],
        priced_utility={},
        violation_magnitude_so_far=0.0,
        latency_seconds=0.0,
        remaining_before=remaining_before,
        lambda_price_before={},
    )


def _seq(cases, initial_budget):
    return DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget",),
        model_names=("a", "b"),
        initial_budget=initial_budget,
        cases=cases,
    )


def test_no_gap_when_policy_matches_hindsight_every_step() -> None:
    cases = [_case(0, {"a": 1.0, "b": 2.0})]
    seq = _seq(cases, {"budget": 100.0})
    decisions = [_decision("b", {"budget": 100.0})]
    hindsight = HindsightResult(
        value=2.0, choices=["b"], exact=True, optimality_gap=0.0, state_count=1
    )
    result = conservation_depletion_split(seq, decisions, hindsight)
    assert result["regret_from_conservation"] == 0.0
    assert result["regret_from_depletion"] == 0.0
    assert result["total_regret"] == 0.0
    assert result["unattributed_regret"] == 0.0


def test_gap_attributed_to_conservation_when_resources_are_ample() -> None:
    # Policy picked the lower-utility model ('a') while budget was still
    # near-full -- not a genuine scarcity event, so this is hoarding.
    cases = [_case(0, {"a": 1.0, "b": 3.0})]
    seq = _seq(cases, {"budget": 100.0})
    decisions = [_decision("a", {"budget": 99.0})]  # ample remaining
    hindsight = HindsightResult(
        value=3.0, choices=["b"], exact=True, optimality_gap=0.0, state_count=1
    )
    result = conservation_depletion_split(seq, decisions, hindsight)
    assert result["regret_from_conservation"] == pytest.approx(2.0)
    assert result["regret_from_depletion"] == 0.0
    assert result["total_regret"] == pytest.approx(2.0)
    assert result["unattributed_regret"] == pytest.approx(0.0)


def test_gap_attributed_to_depletion_when_resources_are_scarce() -> None:
    cases = [_case(0, {"a": 1.0, "b": 3.0})]
    seq = _seq(cases, {"budget": 100.0})
    decisions = [_decision("a", {"budget": 1.0})]  # nearly exhausted (<10% of 100)
    hindsight = HindsightResult(
        value=3.0, choices=["b"], exact=True, optimality_gap=0.0, state_count=1
    )
    result = conservation_depletion_split(seq, decisions, hindsight)
    assert result["regret_from_depletion"] == pytest.approx(2.0)
    assert result["regret_from_conservation"] == 0.0


def test_defer_choices_score_as_zero_utility() -> None:
    cases = [_case(0, {"a": 1.0, "b": 3.0})]
    seq = _seq(cases, {"budget": 100.0})
    decisions = [_decision("defer", {"budget": 99.0})]
    hindsight = HindsightResult(
        value=3.0, choices=["b"], exact=True, optimality_gap=0.0, state_count=1
    )
    result = conservation_depletion_split(seq, decisions, hindsight)
    assert result["total_regret"] == pytest.approx(3.0)
    assert result["regret_from_conservation"] == pytest.approx(3.0)


def test_hindsight_defer_and_policy_choice_never_produces_negative_gap() -> None:
    cases = [_case(0, {"a": 1.0, "b": 3.0})]
    seq = _seq(cases, {"budget": 100.0})
    decisions = [_decision("b", {"budget": 99.0})]  # policy does BETTER than hindsight's "defer"
    hindsight = HindsightResult(
        value=0.0, choices=["defer"], exact=True, optimality_gap=0.0, state_count=1
    )
    result = conservation_depletion_split(seq, decisions, hindsight)
    assert result["regret_from_conservation"] == 0.0
    assert result["regret_from_depletion"] == 0.0
    assert result["total_regret"] == pytest.approx(-3.0)


def test_multi_step_splits_accumulate_across_steps() -> None:
    cases = [
        _case(0, {"a": 1.0, "b": 3.0}),
        _case(1, {"a": 1.0, "b": 3.0}),
    ]
    seq = _seq(cases, {"budget": 100.0})
    decisions = [
        _decision("a", {"budget": 99.0}),  # ample -> conservation
        _decision("a", {"budget": 1.0}),  # scarce -> depletion
    ]
    hindsight = HindsightResult(
        value=6.0, choices=["b", "b"], exact=True, optimality_gap=0.0, state_count=1
    )
    result = conservation_depletion_split(seq, decisions, hindsight)
    assert result["regret_from_conservation"] == pytest.approx(2.0)
    assert result["regret_from_depletion"] == pytest.approx(2.0)
    assert result["total_regret"] == pytest.approx(4.0)
