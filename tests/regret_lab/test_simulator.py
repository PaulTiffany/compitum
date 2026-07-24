"""Online policy simulator -- feasibility gating, reservation/correction
ledger (including delayed revelation producing a real violation), dual
-controller pricing effects, forecaster wiring, and bookkeeping metrics."""

from __future__ import annotations

from compitum.regret_lab.environment import DynamicCase, DynamicSequence
from compitum.regret_lab.forecaster import EWMAForecaster
from compitum.regret_lab.pricing import ReactiveController
from compitum.regret_lab.simulator import simulate_policy


def _seq(cases, initial_budget=None, resource_names=("budget", "quota"), model_names=("a", "b")):
    return DynamicSequence(
        sequence_id="s",
        scenario="hand",
        resource_names=resource_names,
        model_names=model_names,
        initial_budget=initial_budget or {"budget": 10.0, "quota": 10.0},
        cases=cases,
    )


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


def test_static_arm_picks_highest_utility_feasible_model() -> None:
    case = _case(
        0,
        {"a": 1.0, "b": 5.0},
        {"a": {"budget": 1.0, "quota": 0.0}, "b": {"budget": 1.0, "quota": 0.0}},
    )
    seq = _seq([case])
    result, decisions = simulate_policy(seq)
    assert result.choices == ["b"]
    assert result.cumulative_utility == 5.0
    assert decisions[0].feasible_models == ["a", "b"]


def test_infeasible_step_defers_and_counts_as_deferral() -> None:
    case = _case(0, {"a": 1.0}, {"a": {"budget": 100.0, "quota": 0.0}})
    seq = _seq([case], initial_budget={"budget": 1.0, "quota": 1.0}, model_names=("a",))
    result, _ = simulate_policy(seq)
    assert result.choices == ["defer"]
    assert result.deferral_count == 1
    assert result.cumulative_utility == 0.0


def test_avoidable_deferral_when_forecast_wrongly_blocks_a_model() -> None:
    # expected_consumption says infeasible, but realized_consumption (the
    # actual truth) would have fit -- the forecast, not reality, caused it.
    case = DynamicCase(
        step=0,
        base_utility={"a": 1.0},
        expected_consumption={"a": {"budget": 100.0, "quota": 0.0}},
        realized_consumption={"a": {"budget": 1.0, "quota": 0.0}},
        revelation_delay=0,
        replenishment={"budget": 0.0, "quota": 0.0},
    )
    seq = _seq([case], initial_budget={"budget": 5.0, "quota": 5.0}, model_names=("a",))
    result, _ = simulate_policy(seq)
    assert result.choices == ["defer"]
    assert result.avoidable_deferral_count == 1


def test_immediate_revelation_can_cause_a_violation() -> None:
    # expected says cheap (fits), realized is far more expensive than what's
    # left -- corrected immediately (revelation_delay=0) -> a real violation.
    case = DynamicCase(
        step=0,
        base_utility={"a": 1.0},
        expected_consumption={"a": {"budget": 1.0, "quota": 0.0}},
        realized_consumption={"a": {"budget": 10.0, "quota": 0.0}},
        revelation_delay=0,
        replenishment={"budget": 0.0, "quota": 0.0},
    )
    seq = _seq([case], initial_budget={"budget": 2.0, "quota": 2.0}, model_names=("a",))
    result, _ = simulate_policy(seq)
    assert result.choices == ["a"]
    assert result.violation_count == 1
    assert result.violation_magnitude > 0.0


def test_delayed_revelation_defers_the_violation_to_the_correct_step() -> None:
    case0 = DynamicCase(
        step=0,
        base_utility={"a": 1.0},
        expected_consumption={"a": {"budget": 1.0, "quota": 0.0}},
        realized_consumption={"a": {"budget": 10.0, "quota": 0.0}},
        revelation_delay=2,
        replenishment={"budget": 0.0, "quota": 0.0},
    )
    case1 = DynamicCase(
        step=1,
        base_utility={"a": 0.0},
        expected_consumption={"a": {"budget": 0.0, "quota": 0.0}},
        realized_consumption={"a": {"budget": 0.0, "quota": 0.0}},
        revelation_delay=0,
        replenishment={"budget": 0.0, "quota": 0.0},
    )
    seq = _seq([case0, case1], initial_budget={"budget": 2.0, "quota": 2.0}, model_names=("a",))
    result, decisions = simulate_policy(seq)
    # No violation recorded until step 2 (due_step = 0 + 2), which is past
    # the end of this 2-step sequence -- so it never lands at all here.
    assert result.violation_count == 0
    assert decisions[0].violation_magnitude_so_far == 0.0


def test_delayed_revelation_lands_within_a_long_enough_sequence() -> None:
    cases = [
        DynamicCase(
            step=0,
            base_utility={"a": 1.0},
            expected_consumption={"a": {"budget": 1.0, "quota": 0.0}},
            realized_consumption={"a": {"budget": 10.0, "quota": 0.0}},
            revelation_delay=1,
            replenishment={"budget": 0.0, "quota": 0.0},
        ),
        DynamicCase(
            step=1,
            base_utility={"a": 0.0},
            expected_consumption={"a": {"budget": 0.0, "quota": 0.0}},
            realized_consumption={"a": {"budget": 0.0, "quota": 0.0}},
            revelation_delay=0,
            replenishment={"budget": 0.0, "quota": 0.0},
        ),
    ]
    seq = _seq(cases, initial_budget={"budget": 2.0, "quota": 2.0}, model_names=("a",))
    result, _ = simulate_policy(seq)
    assert result.violation_count == 1
    assert result.violation_magnitude > 0.0


def test_route_switch_count() -> None:
    cases = [
        _case(
            0,
            {"a": 5.0, "b": 1.0},
            {"a": {"budget": 1.0, "quota": 0.0}, "b": {"budget": 1.0, "quota": 0.0}},
        ),
        _case(
            1,
            {"a": 1.0, "b": 5.0},
            {"a": {"budget": 1.0, "quota": 0.0}, "b": {"budget": 1.0, "quota": 0.0}},
        ),
        _case(
            2,
            {"a": 1.0, "b": 5.0},
            {"a": {"budget": 1.0, "quota": 0.0}, "b": {"budget": 1.0, "quota": 0.0}},
        ),
    ]
    seq = _seq(cases)
    result, _ = simulate_policy(seq)
    assert result.choices == ["a", "b", "b"]
    assert result.route_switch_count == 1


def test_pricing_controller_changes_the_selected_model() -> None:
    case = _case(
        0,
        {"cheap": 1.0, "rich": 1.5},
        {"cheap": {"budget": 1.0, "quota": 0.0}, "rich": {"budget": 10.0, "quota": 0.0}},
    )
    seq = _seq(
        [case], initial_budget={"budget": 100.0, "quota": 100.0}, model_names=("cheap", "rich")
    )
    without_pricing, _ = simulate_policy(seq)
    assert without_pricing.choices == ["rich"]  # 1.5 > 1.0, no cost penalty

    controller = ReactiveController(resource_names=("budget", "quota"), eta=0.0)
    controller._dual.lambda_price = {"budget": 1.0, "quota": 0.0}
    with_pricing, _ = simulate_policy(seq, pricing_controller=controller)
    # priced: cheap = 1.0 - 1*1.0 = 0.0; rich = 1.5 - 1*10 = -8.5 -> cheap wins
    assert with_pricing.choices == ["cheap"]
    # pricing away from the highest-base-utility model, despite 'rich' being
    # genuinely affordable, is exactly the hoarding signature this tranche
    # added a diagnostic for.
    assert with_pricing.high_value_rejections == 1


def test_forecaster_is_applied_to_feasibility_and_pricing() -> None:
    case = _case(
        0,
        {"a": 1.0},
        {"a": {"budget": 1.0, "quota": 0.0}},
    )
    seq = _seq([case], initial_budget={"budget": 0.5, "quota": 0.5}, model_names=("a",))

    class _AlwaysZeroForecaster:
        def __call__(self, expected, context=None):
            return {m: {r: 0.0 for r in v} for m, v in expected.items()}

    result, _ = simulate_policy(seq, forecaster=_AlwaysZeroForecaster())
    assert result.choices == ["a"]  # forecast says free, so it's feasible despite the real cost 1.0
    assert result.violation_count == 1  # but the correction reveals the truth


def test_forecaster_update_is_called_with_chosen_model_only() -> None:
    case = _case(
        0,
        {"a": 1.0, "b": 1.0},
        {
            "a": {"budget": 1.0, "quota": 0.0},
            "b": {"budget": 1.0, "quota": 0.0},
        },
    )
    seq = _seq([case], model_names=("a", "b"))
    forecaster = EWMAForecaster(alpha=1.0)
    simulate_policy(seq, forecaster=forecaster, forecaster_update=forecaster.update)
    # 'a' beats 'b' alphabetically in a tie (max() keeps first max); only its
    # bias should have been updated (residual 0 either way, but must not crash
    # and must only touch the chosen key).
    assert ("b", "budget") not in forecaster._bias or forecaster._bias[("b", "budget")] == 0.0


def test_depleted_budget_event_recorded() -> None:
    case = _case(0, {"a": 1.0}, {"a": {"budget": 5.0, "quota": 0.0}})
    seq = _seq([case], initial_budget={"budget": 5.0, "quota": 5.0}, model_names=("a",))
    result, _ = simulate_policy(seq)
    assert result.depleted_budget_events == 1


def test_total_consumption_tracks_realized_values_of_chosen_model_only() -> None:
    cases = [
        _case(
            0,
            {"a": 1.0, "b": 5.0},
            {"a": {"budget": 1.0, "quota": 0.0}, "b": {"budget": 2.0, "quota": 1.0}},
        )
    ]
    seq = _seq(cases, initial_budget={"budget": 100.0, "quota": 100.0})
    result, _ = simulate_policy(seq)
    assert result.choices == ["b"]
    assert result.total_consumption == {"budget": 2.0, "quota": 1.0}


def test_decision_to_dict_has_all_fields() -> None:
    case = _case(0, {"a": 1.0}, {"a": {"budget": 1.0, "quota": 0.0}})
    seq = _seq([case], model_names=("a",))
    _, decisions = simulate_policy(seq)
    d = decisions[0].to_dict()
    assert d["chosen"] == "a"
    assert d["feasible_models"] == ["a"]
    assert "priced_utility" in d
    assert "latency_seconds" in d
