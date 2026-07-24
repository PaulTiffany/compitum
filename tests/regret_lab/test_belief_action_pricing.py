"""Bellman-consistent discrete shadow-price action scoring -- Gate A-prime:
the translation-correctness theorem this module must satisfy before any
learning experiment is trusted (per docs/adr/0008)."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.belief_action_pricing import (
    action_shadow_charge,
    run_shadow_charge_policy,
    unit_marginal_prices,
)
from compitum.regret_lab.belief_bellman import BellmanOracle
from compitum.regret_lab.belief_online_optimum import run_online_optimal_policy
from compitum.regret_lab.belief_pricing import ExactBeliefEstimator
from compitum.regret_lab.belief_regime import GRID_UNIT, INITIAL_BELIEF, generate_belief_sequence


def _sequence(seed: int = 0, sequence_id: str = "s0"):
    rng = np.random.default_rng(seed)
    return generate_belief_sequence(rng, sequence_id)


class TestTelescopingIdentity:
    @pytest.mark.parametrize("num_units", [1, 2, 3, 8, 16])
    @pytest.mark.parametrize("belief", [0.1, 0.5, 0.9])
    def test_unit_prices_sum_to_action_shadow_charge(self, num_units: int, belief: float) -> None:
        oracle = BellmanOracle()
        prices = unit_marginal_prices(oracle, 5, 10.0, belief, num_units)
        charge = action_shadow_charge(oracle, 5, 10.0, belief, num_units * GRID_UNIT)
        assert sum(prices) == pytest.approx(charge, abs=1e-9)

    def test_negative_num_units_gives_credit_sequence(self) -> None:
        oracle = BellmanOracle()
        prices = unit_marginal_prices(oracle, 5, 8.0, 0.5, -1)
        charge = action_shadow_charge(oracle, 5, 8.0, 0.5, -1 * GRID_UNIT)
        assert sum(prices) == pytest.approx(charge, abs=1e-9)
        assert len(prices) == 1

    def test_zero_units_gives_empty_sequence_and_zero_charge(self) -> None:
        oracle = BellmanOracle()
        prices = unit_marginal_prices(oracle, 5, 8.0, 0.5, 0)
        assert prices == []
        charge = action_shadow_charge(oracle, 5, 8.0, 0.5, 0.0)
        assert charge == pytest.approx(0.0)


class TestGateAPrimeExactEquivalence:
    """The core correctness requirement: with the exact belief, this
    module's action selection must be BIT-IDENTICAL to the literal
    Bellman-optimal online policy, at every step, on every sequence."""

    @pytest.mark.parametrize("seed", list(range(15)))
    def test_identical_choices_to_online_optimum(self, seed: int) -> None:
        seq, _, _, _ = _sequence(seed=seed, sequence_id=f"s{seed}")
        oracle = BellmanOracle()
        online_result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        shadow_result, _, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        assert shadow_result.choices == online_result.choices

    @pytest.mark.parametrize("seed", list(range(15)))
    def test_identical_cumulative_utility(self, seed: int) -> None:
        seq, _, _, _ = _sequence(seed=seed, sequence_id=f"s{seed}")
        oracle = BellmanOracle()
        online_result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        shadow_result, _, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        assert shadow_result.cumulative_utility == pytest.approx(online_result.cumulative_utility)

    def test_zero_regret_vs_exact_online_optimum(self) -> None:
        seq, _, _, _ = _sequence(seed=3, sequence_id="s3")
        oracle = BellmanOracle()
        online_result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        shadow_result, _, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        regret = online_result.cumulative_utility - shadow_result.cumulative_utility
        assert regret == pytest.approx(0.0)

    def test_zero_violations_and_zero_depletion_events(self) -> None:
        seq, _, _, _ = _sequence(seed=9, sequence_id="s9")
        oracle = BellmanOracle()
        result, _, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        assert result.violation_count == 0
        assert result.violation_magnitude == 0.0
        assert result.depleted_budget_events == 0


class TestBeliefTimingTrace:
    def test_trace_has_one_entry_per_step(self) -> None:
        seq, _, _, _ = _sequence(seed=1, sequence_id="s1")
        oracle = BellmanOracle()
        _, _, traces = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        assert len(traces) == len(seq.cases)

    def test_predicted_next_belief_matches_exact_formula(self) -> None:
        seq, _, belief_priors, _ = _sequence(seed=2, sequence_id="s2")
        oracle = BellmanOracle()
        _, _, traces = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        for t, trace in enumerate(traces):
            assert trace.prior_belief == pytest.approx(belief_priors[t])
            if t + 1 < len(belief_priors):
                assert trace.predicted_next_belief == pytest.approx(belief_priors[t + 1])

    def test_trace_records_distinct_prior_and_next_belief(self) -> None:
        seq, _, _, _ = _sequence(seed=4, sequence_id="s4")
        oracle = BellmanOracle()
        _, _, traces = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        # At least one step should show a genuine prior != next-belief
        # transition (the whole point of the timing audit).
        assert any(t.prior_belief != t.predicted_next_belief for t in traces)

    def test_bellman_q_and_shadow_charge_recorded_per_action(self) -> None:
        seq, _, _, _ = _sequence(seed=5, sequence_id="s5")
        oracle = BellmanOracle()
        _, _, traces = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        for trace in traces:
            assert "defer" in trace.action_shadow_charge
            assert "defer" in trace.bellman_q
            assert trace.selected_action in trace.bellman_q


class TestFeasibilityAndTieBreaking:
    def test_never_selects_unavailable_opportunity(self) -> None:
        seq, _, _, _ = _sequence(seed=6, sequence_id="s6")
        oracle = BellmanOracle()
        result, decisions, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        for case, decision in zip(seq.cases, decisions):
            if decision.chosen == "opportunity":
                assert case.base_utility.get("opportunity", 0.0) > 0.0

    def test_selected_action_always_in_feasible_or_defer(self) -> None:
        seq, _, _, _ = _sequence(seed=7, sequence_id="s7")
        oracle = BellmanOracle()
        _, decisions, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        for decision in decisions:
            assert decision.chosen == "defer" or decision.chosen in decision.feasible_models


class TestRobustnessAcrossManySeeds:
    @pytest.mark.parametrize("seed", list(range(20, 40)))
    def test_exact_equivalence_holds_broadly(self, seed: int) -> None:
        seq, _, _, _ = _sequence(seed=seed, sequence_id=f"wide{seed}")
        oracle = BellmanOracle()
        online_result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        shadow_result, _, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        assert shadow_result.choices == online_result.choices
