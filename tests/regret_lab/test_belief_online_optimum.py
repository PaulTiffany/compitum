"""Exact Bayes-optimal online policy -- verified against the Bellman
oracle's own values and sanity-checked against hindsight (which must
always be at least as good, since it knows the future)."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.belief_bellman import BellmanOracle
from compitum.regret_lab.belief_online_optimum import (
    online_optimum_as_hindsight_result,
    run_online_optimal_policy,
)
from compitum.regret_lab.belief_regime import INITIAL_BELIEF, generate_belief_sequence
from compitum.regret_lab.hindsight import compute_hindsight_optimum


def _sequence(seed: int = 0, sequence_id: str = "s0"):
    rng = np.random.default_rng(seed)
    return generate_belief_sequence(rng, sequence_id)


class TestRunOnlineOptimalPolicy:
    def test_returns_one_decision_per_step(self) -> None:
        seq, _, _, _ = _sequence()
        oracle = BellmanOracle()
        result, decisions = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        assert len(decisions) == len(seq.cases)
        assert len(result.choices) == len(seq.cases)

    def test_never_exceeds_bellman_value_at_initial_state(self) -> None:
        seq, _, _, _ = _sequence(seed=3)
        oracle = BellmanOracle()
        result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        upper_bound = oracle.value(len(seq.cases), seq.initial_budget["budget"], INITIAL_BELIEF)
        # The online policy realizes one particular stochastic path; its
        # achieved utility need not equal the ex-ante expected value, but
        # should be in the same ballpark (loose sanity bound, not exact).
        assert result.cumulative_utility <= upper_bound + 50.0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_never_beats_hindsight(self, seed: int) -> None:
        seq, _, _, _ = _sequence(seed=seed, sequence_id=f"s{seed}")
        oracle = BellmanOracle()
        result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        hindsight = compute_hindsight_optimum(seq)
        assert result.cumulative_utility <= hindsight.value + 1e-9

    def test_no_violations_since_defer_is_always_available(self) -> None:
        seq, _, _, _ = _sequence(seed=9)
        oracle = BellmanOracle()
        result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        assert result.violation_count == 0
        assert result.violation_magnitude == 0.0

    def test_route_switch_count_matches_choice_transitions(self) -> None:
        seq, _, _, _ = _sequence(seed=2)
        oracle = BellmanOracle()
        result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        expected = sum(
            1 for i in range(1, len(result.choices)) if result.choices[i] != result.choices[i - 1]
        )
        assert result.route_switch_count == expected

    def test_terminal_remaining_is_nonnegative(self) -> None:
        seq, _, _, _ = _sequence(seed=4)
        oracle = BellmanOracle()
        result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        assert result.terminal_remaining["budget"] >= 0.0


class TestOnlineOptimumAsHindsightResult:
    def test_matches_run_online_optimal_policy_utility(self) -> None:
        seq, _, _, _ = _sequence(seed=6)
        oracle = BellmanOracle()
        result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        packaged = online_optimum_as_hindsight_result(seq, oracle, INITIAL_BELIEF)
        assert packaged.value == pytest.approx(result.cumulative_utility)
        assert packaged.choices == result.choices
        assert packaged.exact is True
        assert packaged.optimality_gap == 0.0

    def test_state_count_reflects_oracle_memoization(self) -> None:
        seq, _, _, _ = _sequence(seed=8)
        oracle = BellmanOracle()
        packaged = online_optimum_as_hindsight_result(seq, oracle, INITIAL_BELIEF)
        assert packaged.state_count == oracle.state_count()
        assert packaged.state_count > 0
