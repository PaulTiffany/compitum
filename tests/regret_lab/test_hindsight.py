"""Hindsight constrained oracle -- exact on hand-constructed cases with
known answers, plus the bounded-quality fallback path."""

from __future__ import annotations

from compitum.regret_lab.environment import DynamicCase, DynamicSequence
from compitum.regret_lab.hindsight import compute_hindsight_optimum


def _case(step, base_utility, consumption, replenishment=None, revelation_delay=0):
    replenishment = replenishment or {"budget": 0.0, "quota": 0.0}
    return DynamicCase(
        step=step,
        base_utility=base_utility,
        expected_consumption=consumption,
        realized_consumption=consumption,
        revelation_delay=revelation_delay,
        replenishment=replenishment,
    )


def test_single_step_picks_the_only_affordable_model() -> None:
    case = _case(
        0,
        {"cheap": 1.0, "rich": 10.0},
        {
            "cheap": {"budget": 1.0, "quota": 0.0},
            "rich": {"budget": 100.0, "quota": 0.0},
        },
    )
    seq = DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget", "quota"),
        model_names=("cheap", "rich"),
        initial_budget={"budget": 5.0, "quota": 0.0},
        cases=[case],
    )
    result = compute_hindsight_optimum(seq)
    assert result.exact is True
    assert result.value == 1.0
    assert result.choices == ["cheap"]
    assert result.optimality_gap == 0.0


def test_defer_is_correct_when_nothing_is_affordable() -> None:
    case = _case(0, {"only": 5.0}, {"only": {"budget": 10.0, "quota": 0.0}})
    seq = DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget", "quota"),
        model_names=("only",),
        initial_budget={"budget": 1.0, "quota": 0.0},
        cases=[case],
    )
    result = compute_hindsight_optimum(seq)
    assert result.value == 0.0
    assert result.choices == ["defer"]


def test_conserving_now_enables_a_better_future_choice() -> None:
    # Step 0: 'ok' (utility 1, cost 1) vs 'great' (utility 2, cost 3) -- both
    # affordable with budget=3. Step 1: 'jackpot' costs 2 and pays 100, but
    # only reachable if step 0 conserved (picked 'ok' or deferred).
    cases = [
        _case(
            0,
            {"ok": 1.0, "great": 2.0},
            {
                "ok": {"budget": 1.0, "quota": 0.0},
                "great": {"budget": 3.0, "quota": 0.0},
            },
        ),
        _case(1, {"jackpot": 100.0}, {"jackpot": {"budget": 2.0, "quota": 0.0}}),
    ]
    seq = DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget", "quota"),
        model_names=("ok", "great", "jackpot"),
        initial_budget={"budget": 3.0, "quota": 0.0},
        cases=cases,
    )
    result = compute_hindsight_optimum(seq)
    # Greedy (picking 'great' at step 0 for its higher immediate utility)
    # would leave 0 budget and miss the jackpot entirely (value 2.0).
    # The true hindsight optimum conserves at step 0 to reach the jackpot.
    assert result.value == 1.0 + 100.0
    assert result.choices == ["ok", "jackpot"]


def test_replenishment_makes_a_later_expensive_choice_reachable() -> None:
    cases = [
        _case(
            0,
            {"a": 1.0},
            {"a": {"budget": 1.0, "quota": 0.0}},
            replenishment={"budget": 5.0, "quota": 0.0},
        ),
        _case(1, {"b": 3.0}, {"b": {"budget": 4.0, "quota": 0.0}}),
    ]
    seq = DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget", "quota"),
        model_names=("a", "b"),
        initial_budget={"budget": 1.0, "quota": 0.0},
        cases=cases,
    )
    result = compute_hindsight_optimum(seq)
    assert result.value == 1.0 + 3.0
    assert result.choices == ["a", "b"]


def test_two_resources_must_both_be_satisfied() -> None:
    case = _case(
        0,
        {"m": 5.0},
        {"m": {"budget": 1.0, "quota": 10.0}},
    )
    seq = DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget", "quota"),
        model_names=("m",),
        initial_budget={"budget": 10.0, "quota": 1.0},  # quota insufficient
        cases=[case],
    )
    result = compute_hindsight_optimum(seq)
    assert result.value == 0.0
    assert result.choices == ["defer"]


def test_fallback_path_triggers_and_reports_a_gap() -> None:
    # A long, richly-branching sequence with a tiny max_states forces the
    # fallback; the reported result must be internally consistent (greedy
    # value, exact=False, and a nonnegative gap against the trivial
    # per-step-max upper bound) rather than silently pretending exactness.
    cases = [
        _case(
            t,
            {"a": 1.0 + t * 0.1, "b": 2.0 - t * 0.05, "c": 1.5},
            {
                "a": {"budget": 1.0, "quota": 0.5},
                "b": {"budget": 0.75, "quota": 0.75},
                "c": {"budget": 0.5, "quota": 1.0},
            },
            replenishment={"budget": 0.9, "quota": 0.9},
        )
        for t in range(20)
    ]
    seq = DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget", "quota"),
        model_names=("a", "b", "c"),
        initial_budget={"budget": 5.0, "quota": 5.0},
        cases=cases,
    )
    result = compute_hindsight_optimum(seq, max_states=50)
    assert result.exact is False
    assert result.optimality_gap >= 0.0
    upper_bound = sum(max(c.base_utility.values()) for c in cases)
    assert result.value <= upper_bound + 1e-9


def test_greedy_fallback_defers_when_nothing_is_affordable() -> None:
    # Same branchy shape as the fallback test above (forces max_states to be
    # exceeded), but with one step (t=5) so expensive that no model can
    # possibly afford it regardless of history -- the greedy fallback
    # itself must defer on that step, not just pick among options.
    cases = []
    for t in range(20):
        base_utility = {"a": 1.0 + t * 0.1, "b": 2.0 - t * 0.05, "c": 1.5}
        consumption = {
            "a": {"budget": 1.0, "quota": 0.5},
            "b": {"budget": 0.75, "quota": 0.75},
            "c": {"budget": 0.5, "quota": 1.0},
        }
        if t == 5:
            consumption = {m: {"budget": 1000.0, "quota": 1000.0} for m in consumption}
        cases.append(
            _case(t, base_utility, consumption, replenishment={"budget": 0.9, "quota": 0.9})
        )
    seq = DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget", "quota"),
        model_names=("a", "b", "c"),
        initial_budget={"budget": 5.0, "quota": 5.0},
        cases=cases,
    )
    result = compute_hindsight_optimum(seq, max_states=50)
    assert result.exact is False
    assert result.choices[5] == "defer"


def test_to_dict_reports_all_fields() -> None:
    case = _case(0, {"a": 1.0}, {"a": {"budget": 1.0, "quota": 0.0}})
    seq = DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=("budget", "quota"),
        model_names=("a",),
        initial_budget={"budget": 5.0, "quota": 5.0},
        cases=[case],
    )
    result = compute_hindsight_optimum(seq)
    d = result.to_dict()
    assert d == {
        "value": 1.0,
        "choices": ["a"],
        "exact": True,
        "optimality_gap": 0.0,
        "state_count": d["state_count"],
    }
    assert d["state_count"] >= 1
