"""Shadow-charge routing for the belief-sensitive environment (tranche
7) -- same Gate A-prime exact-equivalence requirement as tranche 6.5,
plus a dedicated regression test for the one thing that could not be
reused unchanged: "opportunity" must be scored by belief-weighted
expectation, never by its true realized (hidden) payoff."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.belief_action_pricing_v2 import (
    ExactBeliefEstimatorV2,
    HmmBeliefEstimatorV2,
    run_shadow_charge_policy_v2,
)
from compitum.regret_lab.belief_bellman_v2 import BeliefSensitiveBellmanOracle
from compitum.regret_lab.belief_online_optimum import run_online_optimal_policy
from compitum.regret_lab.belief_pricing import ExactBeliefEstimator, LookupBeliefEstimator
from compitum.regret_lab.belief_regime import INITIAL_BELIEF
from compitum.regret_lab.belief_regime_v2 import generate_belief_sequence_v2
from compitum.regret_lab.pricing import PricingUpdateContext

U_NORMAL = 1.0
U_HIGH = 8.0


def _decision_context(case, step: int, total_steps: int) -> PricingUpdateContext:
    return PricingUpdateContext(
        resource_names=("budget",),
        reservation={},
        remaining_before={"budget": 6.0},
        remaining_after={"budget": 6.0},
        step=step,
        total_steps=total_steps,
        case=case,
        chosen="defer",
    )


def _sequence(seed: int, sequence_id: str, initial_budget: float = 8.0):
    rng = np.random.default_rng(seed)
    return generate_belief_sequence_v2(
        rng, sequence_id, initial_budget=initial_budget, u_normal=U_NORMAL, u_high=U_HIGH
    )


class TestGateAPrimeV2ExactEquivalence:
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_identical_choices_to_online_optimum(self, seed: int) -> None:
        seq, _, _, _ = _sequence(seed=seed, sequence_id=f"s{seed}")
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH)
        online_result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        shadow_result, _, _ = run_shadow_charge_policy_v2(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF), u_normal=U_NORMAL, u_high=U_HIGH
        )
        assert shadow_result.choices == online_result.choices

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_identical_cumulative_utility(self, seed: int) -> None:
        seq, _, _, _ = _sequence(seed=seed, sequence_id=f"s{seed}")
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH)
        online_result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        shadow_result, _, _ = run_shadow_charge_policy_v2(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF), u_normal=U_NORMAL, u_high=U_HIGH
        )
        assert shadow_result.cumulative_utility == pytest.approx(online_result.cumulative_utility)

    def test_zero_violations_and_depletion(self) -> None:
        seq, _, _, _ = _sequence(seed=3, sequence_id="s3")
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH)
        result, _, _ = run_shadow_charge_policy_v2(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF), u_normal=U_NORMAL, u_high=U_HIGH
        )
        assert result.violation_count == 0
        assert result.violation_magnitude == 0.0
        assert result.depleted_budget_events == 0


class TestScoringUsesExpectationNotTruth:
    """The one thing that genuinely could not be reused unchanged from
    tranche 6.5: verifying it directly, since a regression here would
    silently leak the hidden regime into the decision."""

    def test_low_fixed_belief_never_chooses_opportunity_even_if_true_regime_is_high(self) -> None:
        # Construct a sequence where the true regime is HIGH (opportunity
        # truly worth U_HIGH=8.0 if taken) but feed the policy a FIXED,
        # confidently-wrong belief near 0 (posterior stays near 0). If
        # the scoring function ever leaked the true realized utility, it
        # would grab "opportunity" (8.0 far exceeds spend's 2.0); if it
        # correctly scores by the belief-weighted expectation, the low
        # belief makes opportunity look worth ~U_NORMAL=1.0, losing to
        # spend, regardless of what the hidden regime actually is.
        for seed in range(30):
            rng = np.random.default_rng(seed)
            seq, true_regimes, _, _ = generate_belief_sequence_v2(
                rng, f"biased{seed}", initial_budget=8.0, u_normal=U_NORMAL, u_high=U_HIGH
            )
            if not any(
                r == 1 and c.base_utility["opportunity"] > 0.0
                for r, c in zip(true_regimes, seq.cases)
            ):
                continue  # need a sequence where a true-HIGH opportunity actually appears
            oracle = BeliefSensitiveBellmanOracle(
                u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH
            )
            fixed_low = LookupBeliefEstimator(beliefs=[0.0] * len(seq.cases), initial_belief=0.0)
            result, decisions, _ = run_shadow_charge_policy_v2(
                seq, oracle, fixed_low, u_normal=U_NORMAL, u_high=U_HIGH
            )
            for r, c, d in zip(true_regimes, seq.cases, decisions):
                if r == 1 and c.base_utility["opportunity"] > 0.0:
                    assert d.chosen != "opportunity"
            return
        pytest.skip("no seed in range produced a true-HIGH opportunity appearance")


class TestFeasibilityAndTieBreaking:
    def test_never_selects_unavailable_opportunity(self) -> None:
        seq, _, _, _ = _sequence(seed=6, sequence_id="s6")
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH)
        _, decisions, _ = run_shadow_charge_policy_v2(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF), u_normal=U_NORMAL, u_high=U_HIGH
        )
        for case, decision in zip(seq.cases, decisions):
            if decision.chosen == "opportunity":
                assert case.realized_consumption["opportunity"]["budget"] < 1e5

    def test_selected_action_always_feasible_or_defer(self) -> None:
        seq, _, _, _ = _sequence(seed=8, sequence_id="s8")
        oracle = BeliefSensitiveBellmanOracle(u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH)
        _, decisions, _ = run_shadow_charge_policy_v2(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF), u_normal=U_NORMAL, u_high=U_HIGH
        )
        for decision in decisions:
            assert decision.chosen == "defer" or decision.chosen in decision.feasible_models


class TestHighValueRejectionIsLive:
    def test_can_be_nonzero_with_a_deliberately_wrong_belief(self) -> None:
        found_nonzero = False
        for seed in range(40):
            seq, _, _, _ = _sequence(seed=seed, sequence_id=f"hv{seed}")
            oracle = BeliefSensitiveBellmanOracle(
                u_normal_opportunity=U_NORMAL, u_high_opportunity=U_HIGH
            )
            inverted = LookupBeliefEstimator(beliefs=[1.0] * len(seq.cases), initial_belief=1.0)
            result, _, _ = run_shadow_charge_policy_v2(
                seq, oracle, inverted, u_normal=U_NORMAL, u_high=U_HIGH
            )
            if result.high_value_rejections > 0:
                found_nonzero = True
                break
        assert found_nonzero


class TestExactAndHmmEstimatorV2CrossValidation:
    """Independently-coded (scalar closed-form vs. generic matrix)
    filters of the same parameterized dynamics -- must agree exactly,
    mirroring tranche 6's own ExactBeliefEstimator/HmmBeliefEstimator
    cross-check."""

    def test_initial_belief_matches(self) -> None:
        exact = ExactBeliefEstimatorV2(belief=0.3)
        hmm = HmmBeliefEstimatorV2(belief_vector=np.array([0.7, 0.3]))
        assert hmm.current_belief() == pytest.approx(exact.current_belief())

    def test_agrees_over_a_multi_step_trajectory(self) -> None:
        seq, _, _, _ = _sequence(seed=11, sequence_id="cross11")
        exact = ExactBeliefEstimatorV2(belief=INITIAL_BELIEF)
        hmm = HmmBeliefEstimatorV2(belief_vector=np.array([1.0 - INITIAL_BELIEF, INITIAL_BELIEF]))
        for t, case in enumerate(seq.cases):
            assert hmm.current_belief() == pytest.approx(exact.current_belief(), abs=1e-9)
            ctx = _decision_context(case, step=t, total_steps=len(seq.cases))
            exact.advance(ctx)
            hmm.advance(ctx)
        assert hmm.current_belief() == pytest.approx(exact.current_belief(), abs=1e-9)

    def test_agrees_at_tuned_parameters(self) -> None:
        p_normal, p_high, t_n2h, t_h2h = 0.15, 0.25, 0.3, 0.85
        rng = np.random.default_rng(21)
        seq, _, _, _ = generate_belief_sequence_v2(
            rng,
            "cross-tuned",
            initial_budget=6.0,
            u_normal=U_NORMAL,
            u_high=U_HIGH,
            p_opportunity_normal=p_normal,
            p_opportunity_high=p_high,
            transition_normal_to_high=t_n2h,
            transition_high_to_high=t_h2h,
        )
        exact = ExactBeliefEstimatorV2(
            belief=INITIAL_BELIEF,
            p_opportunity_normal=p_normal,
            p_opportunity_high=p_high,
            transition_normal_to_high=t_n2h,
            transition_high_to_high=t_h2h,
        )
        hmm = HmmBeliefEstimatorV2(
            belief_vector=np.array([1.0 - INITIAL_BELIEF, INITIAL_BELIEF]),
            p_opportunity_normal=p_normal,
            p_opportunity_high=p_high,
            transition_normal_to_high=t_n2h,
            transition_high_to_high=t_h2h,
        )
        for t, case in enumerate(seq.cases):
            ctx = _decision_context(case, step=t, total_steps=len(seq.cases))
            exact.advance(ctx)
            hmm.advance(ctx)
        assert hmm.current_belief() == pytest.approx(exact.current_belief(), abs=1e-9)
