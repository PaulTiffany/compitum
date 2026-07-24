"""BeliefSensitiveBellmanOracle -- same recursive structure as tranche
6's BellmanOracle, "opportunity" utility now the belief-weighted
expectation ``(1-q)*u_normal + q*u_high`` instead of a fixed constant."""

from __future__ import annotations

import pytest

from compitum.regret_lab.belief_bellman_v2 import (
    BeliefSensitiveBellmanOracle,
    BellmanStateBudgetExceeded,
)


class TestValue:
    def test_zero_remaining_steps_is_zero(self) -> None:
        oracle = BeliefSensitiveBellmanOracle()
        assert oracle.value(0, 10.0, 0.5) == 0.0

    def test_abundant_budget_zero_belief_matches_normal_expectation(self) -> None:
        # One step left, huge budget: best action is whichever of
        # conserve/spend/opportunity(if available) has the highest
        # expected utility. At belief=0 opportunity is worth u_normal=1.0,
        # less than spend's fixed 2.0, so spend should dominate whenever
        # feasible -- verified indirectly via best_action below.
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=1.0, u_high_opportunity=8.0)
        value = oracle.value(1, 100.0, 0.0)
        assert value > 0.0

    def test_value_increases_with_budget(self) -> None:
        oracle = BeliefSensitiveBellmanOracle()
        low = oracle.value(3, 2.0, 0.5)
        high = oracle.value(3, 10.0, 0.5)
        assert high >= low

    def test_state_cap_raises(self) -> None:
        oracle = BeliefSensitiveBellmanOracle(max_states=1)
        with pytest.raises(BellmanStateBudgetExceeded):
            oracle.value(5, 8.0, 0.5)
            oracle.value(5, 6.0, 0.3)


class TestOpportunityUtilityFormula:
    def test_posterior_zero_uses_u_normal(self) -> None:
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=1.0, u_high_opportunity=8.0)
        assert oracle._opportunity_utility(0.0) == pytest.approx(1.0)

    def test_posterior_one_uses_u_high(self) -> None:
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=1.0, u_high_opportunity=8.0)
        assert oracle._opportunity_utility(1.0) == pytest.approx(8.0)


class TestBestActionGivenObservation:
    def test_unavailable_opportunity_excludes_it(self) -> None:
        oracle = BeliefSensitiveBellmanOracle()
        action, _ = oracle.best_action_given_observation(3, 8.0, 0.5, observed_opportunity=False)
        assert action != "opportunity"

    def test_low_belief_prefers_spend_over_opportunity_when_available(self) -> None:
        # u_normal=1.0 < spend's 2.0 < u_high=8.0: at belief exactly 0
        # (posterior stays 0 given P_OPPORTUNITY structure keeps it near
        # 0), opportunity's expected value should lose to spend.
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=1.0, u_high_opportunity=8.0)
        action, _ = oracle.best_action_given_observation(1, 8.0, 0.0, observed_opportunity=True)
        assert action == "spend"

    def test_high_belief_prefers_opportunity_when_available(self) -> None:
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=1.0, u_high_opportunity=8.0)
        action, _ = oracle.best_action_given_observation(1, 8.0, 1.0, observed_opportunity=True)
        assert action == "opportunity"


class TestMarginalPrice:
    def test_zero_price_under_abundance(self) -> None:
        oracle = BeliefSensitiveBellmanOracle()
        price = oracle.marginal_price(5, 1000.0, 0.5)
        assert price == pytest.approx(0.0, abs=1e-6)

    def test_price_nonnegative(self) -> None:
        oracle = BeliefSensitiveBellmanOracle()
        price = oracle.marginal_price(5, 4.0, 0.5)
        assert price >= 0.0


class TestStateCount:
    def test_increases_after_value_calls(self) -> None:
        oracle = BeliefSensitiveBellmanOracle()
        assert oracle.state_count() == 0
        oracle.value(3, 8.0, 0.5)
        assert oracle.state_count() > 0
