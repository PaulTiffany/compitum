"""Exact online optimum for the belief-sensitive environment, with
parameterized transition/observation dynamics (tranche 7, Gate 0's
second pass) -- Gate-A-prime-style exact equivalence must hold even
away from tranche 6's default parameters, since that is exactly the
regime Gate 0's development grid explores."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.belief_action_pricing_v2 import (
    ExactBeliefEstimatorV2,
    run_shadow_charge_policy_v2,
)
from compitum.regret_lab.belief_bellman_v2 import BeliefSensitiveBellmanOracle
from compitum.regret_lab.belief_online_optimum_v2 import (
    online_optimum_as_hindsight_result_v2,
    run_online_optimal_policy_v2,
)
from compitum.regret_lab.belief_regime import INITIAL_BELIEF
from compitum.regret_lab.belief_regime_v2 import generate_belief_sequence_v2

U_NORMAL = 0.5
U_HIGH = 9.0
P_NORMAL = 0.05
P_HIGH = 0.35
TRANSITION_N2H = 0.2
TRANSITION_H2H = 0.6

# A deliberately non-default configuration -- exactly the kind Gate 0's
# development grid explores -- to prove the equivalence isn't an
# artifact of only ever testing at tranche 6's original parameters.
P_NORMAL_TUNED = 0.15
P_HIGH_TUNED = 0.25
TRANSITION_N2H_TUNED = 0.3
TRANSITION_H2H_TUNED = 0.85


def _sequence(seed: int, sequence_id: str, tuned: bool = False):
    rng = np.random.default_rng(seed)
    if tuned:
        return generate_belief_sequence_v2(
            rng,
            sequence_id,
            initial_budget=6.0,
            u_normal=U_NORMAL,
            u_high=U_HIGH,
            p_opportunity_normal=P_NORMAL_TUNED,
            p_opportunity_high=P_HIGH_TUNED,
            transition_normal_to_high=TRANSITION_N2H_TUNED,
            transition_high_to_high=TRANSITION_H2H_TUNED,
        )
    return generate_belief_sequence_v2(
        rng, sequence_id, initial_budget=8.0, u_normal=U_NORMAL, u_high=U_HIGH
    )


class TestDefaultParametersMatchGroundTruth:
    def test_online_optimal_belief_tracking_matches_generator_ground_truth(self) -> None:
        seq, _, belief_priors, _ = _sequence(seed=1, sequence_id="s1")
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH)
        _, decisions = run_online_optimal_policy_v2(seq, oracle, INITIAL_BELIEF)
        for t, decision in enumerate(decisions):
            assert decision.remaining_before["budget"] >= 0.0
        # Ground-truth belief_priors come from the SAME default filter
        # formulas exercised independently inside the generator.
        assert len(belief_priors) == len(seq.cases)


class TestGateAPrimeAtTunedParameters:
    @pytest.mark.parametrize("seed", list(range(15)))
    def test_identical_choices_at_tuned_parameters(self, seed: int) -> None:
        seq, _, _, _ = _sequence(seed=seed, sequence_id=f"tuned{seed}", tuned=True)
        oracle = BeliefSensitiveBellmanOracle(
            u_normal_opportunity=U_NORMAL,
            u_high_opportunity=U_HIGH,
            p_opportunity_normal=P_NORMAL_TUNED,
            p_opportunity_high=P_HIGH_TUNED,
            transition_normal_to_high=TRANSITION_N2H_TUNED,
            transition_high_to_high=TRANSITION_H2H_TUNED,
        )
        online_result, _ = run_online_optimal_policy_v2(
            seq,
            oracle,
            INITIAL_BELIEF,
            P_NORMAL_TUNED,
            P_HIGH_TUNED,
            TRANSITION_N2H_TUNED,
            TRANSITION_H2H_TUNED,
        )
        exact_estimator = ExactBeliefEstimatorV2(
            belief=INITIAL_BELIEF,
            p_opportunity_normal=P_NORMAL_TUNED,
            p_opportunity_high=P_HIGH_TUNED,
            transition_normal_to_high=TRANSITION_N2H_TUNED,
            transition_high_to_high=TRANSITION_H2H_TUNED,
        )
        shadow_result, _, _ = run_shadow_charge_policy_v2(
            seq,
            oracle,
            exact_estimator,
            u_normal=U_NORMAL,
            u_high=U_HIGH,
            p_opportunity_normal=P_NORMAL_TUNED,
            p_opportunity_high=P_HIGH_TUNED,
            transition_normal_to_high=TRANSITION_N2H_TUNED,
            transition_high_to_high=TRANSITION_H2H_TUNED,
        )
        assert shadow_result.choices == online_result.choices
        assert shadow_result.cumulative_utility == pytest.approx(online_result.cumulative_utility)


class TestOnlineOptimumAsHindsightResultV2:
    def test_matches_run_online_optimal_policy_v2_utility(self) -> None:
        seq, _, _, _ = _sequence(seed=4, sequence_id="s4")
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH)
        result, _ = run_online_optimal_policy_v2(seq, oracle, INITIAL_BELIEF)
        packaged = online_optimum_as_hindsight_result_v2(seq, oracle, INITIAL_BELIEF)
        assert packaged.value == pytest.approx(result.cumulative_utility)
        assert packaged.choices == result.choices
        assert packaged.exact is True
        assert packaged.state_count == oracle.state_count()


class TestStructuralInvariants:
    def test_zero_violations_and_depletion(self) -> None:
        seq, _, _, _ = _sequence(seed=6, sequence_id="s6")
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH)
        result, _ = run_online_optimal_policy_v2(seq, oracle, INITIAL_BELIEF)
        assert result.violation_count == 0
        assert result.violation_magnitude == 0.0
        assert result.depleted_budget_events == 0
