"""Exact belief-state Bellman oracle -- hand-verified values, prices, and
optimal actions; monotonicity and state-cap safety checks."""

from __future__ import annotations

import pytest

from compitum.regret_lab.belief_bellman import BellmanOracle, BellmanStateBudgetExceeded


def test_zero_remaining_steps_has_zero_value() -> None:
    oracle = BellmanOracle()
    assert oracle.value(0, 100.0, 0.5) == 0.0


def test_one_step_value_matches_hand_computation_ample_budget() -> None:
    # P(o=1|0.5)=0.2, P(o=0)=0.8; ample budget affords everything.
    # V = 0.8*max(conserve=1,spend=2) + 0.2*max(conserve=1,spend=2,opportunity=8)
    #   = 0.8*2 + 0.2*8 = 1.6 + 1.6 = 3.2
    oracle = BellmanOracle()
    assert oracle.value(1, 10.0, 0.5) == pytest.approx(3.2)


def test_one_step_value_with_budget_affording_nothing() -> None:
    oracle = BellmanOracle()
    assert oracle.value(1, 0.5, 0.5) == pytest.approx(0.0)


def test_one_step_value_affording_only_conserve() -> None:
    oracle = BellmanOracle()
    # Only 'conserve' (cost 1.0) fits; best affordable action is the same
    # regardless of observation, so value is just its utility.
    assert oracle.value(1, 1.0, 0.5) == pytest.approx(1.0)


def test_marginal_price_is_zero_when_already_abundant() -> None:
    oracle = BellmanOracle()
    # Both budget=10 and budget=9.5 comfortably afford every action.
    assert oracle.marginal_price(1, 10.0, 0.5) == pytest.approx(0.0)


def test_marginal_price_spikes_at_the_exact_scarcity_boundary() -> None:
    oracle = BellmanOracle()
    # At budget=4.0 the opportunity (cost 4.0) is exactly affordable; at
    # 3.5 it is not -- hand-computed values from the docstring example.
    v_at = oracle.value(1, 4.0, 0.5)
    v_below = oracle.value(1, 3.5, 0.5)
    assert v_at == pytest.approx(3.2)
    assert v_below == pytest.approx(2.0)
    price = oracle.marginal_price(1, 4.0, 0.5)
    assert price == pytest.approx((v_at - v_below) / 0.5)
    assert price > 0.0


def test_marginal_price_increases_with_belief_in_high_regime() -> None:
    oracle = BellmanOracle()
    price_low = oracle.marginal_price(1, 4.0, 0.1)
    price_high = oracle.marginal_price(1, 4.0, 0.9)
    assert price_high > price_low


def test_marginal_price_never_negative() -> None:
    oracle = BellmanOracle()
    for budget in (0.0, 1.0, 2.0, 4.0, 8.0, 20.0):
        for belief in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert oracle.marginal_price(3, budget, belief) >= -1e-9


def test_marginal_price_at_zero_budget_does_not_query_negative_budget() -> None:
    # budget - delta would be negative; must clamp rather than recurse
    # into a nonsensical state.
    oracle = BellmanOracle()
    price = oracle.marginal_price(2, 0.0, 0.5)
    assert price >= 0.0


def test_value_is_monotonically_nondecreasing_in_budget() -> None:
    oracle = BellmanOracle()
    budgets = [0.0, 1.0, 2.0, 4.0, 6.0, 10.0]
    values = [oracle.value(3, b, 0.5) for b in budgets]
    for earlier, later in zip(values, values[1:]):
        assert later >= earlier - 1e-9


def test_best_action_given_observation_picks_opportunity_when_available_and_affordable() -> None:
    oracle = BellmanOracle()
    action, value = oracle.best_action_given_observation(1, 10.0, 0.5, observed_opportunity=True)
    assert action == "opportunity"
    assert value == pytest.approx(8.0)


def test_best_action_given_observation_never_picks_opportunity_when_unavailable() -> None:
    oracle = BellmanOracle()
    action, value = oracle.best_action_given_observation(1, 10.0, 0.5, observed_opportunity=False)
    assert action == "spend"
    assert value == pytest.approx(2.0)


def test_best_action_given_observation_defers_when_nothing_affordable() -> None:
    oracle = BellmanOracle()
    action, value = oracle.best_action_given_observation(1, 0.5, 0.5, observed_opportunity=False)
    assert action == "defer"
    assert value == pytest.approx(0.0)


def test_multi_step_value_runs_and_state_count_is_bounded() -> None:
    oracle = BellmanOracle()
    value = oracle.value(5, 6.0, 0.5)
    assert value > 0.0
    assert oracle.state_count() > 0
    assert oracle.state_count() < 10_000


def test_state_cap_exceeded_raises() -> None:
    oracle = BellmanOracle(max_states=1)
    with pytest.raises(BellmanStateBudgetExceeded):
        oracle.value(5, 6.0, 0.5)


def test_shared_oracle_reuses_memo_across_calls() -> None:
    oracle = BellmanOracle()
    oracle.value(4, 8.0, 0.5)
    first_count = oracle.state_count()
    oracle.value(4, 8.0, 0.5)
    assert oracle.state_count() == first_count  # fully memoized, no new states
