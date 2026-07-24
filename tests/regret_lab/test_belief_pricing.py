"""Belief-estimator-agnostic pricing controller -- verified against the
exact Bellman oracle and cross-checked estimator-by-estimator."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.belief_bellman import BellmanOracle
from compitum.regret_lab.belief_channels import BeliefChannelHistory
from compitum.regret_lab.belief_hmm_filter import belief_high_from_scalar
from compitum.regret_lab.belief_pricing import (
    BeliefPricingController,
    ExactBeliefEstimator,
    HmmBeliefEstimator,
    LookupBeliefEstimator,
    RidgeBeliefEstimator,
    build_belief_training_pairs,
)
from compitum.regret_lab.belief_regime import (
    INITIAL_BELIEF,
    generate_belief_sequence,
)
from compitum.regret_lab.pricing import PricingUpdateContext
from compitum.regret_lab.simulator import simulate_policy
from compitum.regret_lab.windowed_predictor import fit_ridge


def _sequence(sequence_id: str = "s0", seed: int = 0):
    rng = np.random.default_rng(seed)
    return generate_belief_sequence(rng, sequence_id)


def _context(case, chosen="conserve", step=0, total_steps=10, remaining_before=6.0, remaining_after=5.5):
    return PricingUpdateContext(
        resource_names=("budget",),
        reservation={"budget": 1.0},
        remaining_before={"budget": remaining_before},
        remaining_after={"budget": remaining_after},
        step=step,
        total_steps=total_steps,
        case=case,
        chosen=chosen,
    )


class TestExactBeliefEstimator:
    def test_initial_belief_defaults(self) -> None:
        est = ExactBeliefEstimator()
        assert est.current_belief() == INITIAL_BELIEF

    def test_advance_matches_hand_filter(self) -> None:
        seq, _, belief_priors, _ = _sequence()
        est = ExactBeliefEstimator(belief=belief_priors[0])
        ctx = _context(seq.cases[0])
        est.advance(ctx)
        assert est.current_belief() == pytest.approx(belief_priors[1])

    def test_advance_over_full_sequence_matches_ground_truth(self) -> None:
        seq, _, belief_priors, _ = _sequence(seed=7)
        est = ExactBeliefEstimator(belief=belief_priors[0])
        for t, case in enumerate(seq.cases):
            assert est.current_belief() == pytest.approx(belief_priors[t])
            ctx = _context(case, step=t)
            est.advance(ctx)


class TestHmmBeliefEstimator:
    def test_initial_belief_defaults(self) -> None:
        est = HmmBeliefEstimator()
        assert est.current_belief() == pytest.approx(INITIAL_BELIEF)

    def test_matches_exact_estimator_over_full_sequence(self) -> None:
        seq, _, belief_priors, _ = _sequence(seed=3)
        exact = ExactBeliefEstimator(belief=belief_priors[0])
        hmm = HmmBeliefEstimator(belief_vector=belief_high_from_scalar(belief_priors[0]))
        for t, case in enumerate(seq.cases):
            assert hmm.current_belief() == pytest.approx(exact.current_belief(), abs=1e-9)
            ctx = _context(case, step=t)
            exact.advance(ctx)
            hmm.advance(ctx)
        assert hmm.current_belief() == pytest.approx(exact.current_belief(), abs=1e-9)


class TestRidgeBeliefEstimator:
    def test_current_belief_before_any_advance_is_initial(self) -> None:
        model = fit_ridge([[0.0] * 11], [0.5])
        est = RidgeBeliefEstimator(model=model, max_window=1)
        assert est.current_belief() == INITIAL_BELIEF

    def test_advance_updates_belief_and_history(self) -> None:
        features = [[float(i)] * 55 for i in range(5)]
        targets = [0.1, 0.3, 0.5, 0.7, 0.9]
        model = fit_ridge(features, targets)
        est = RidgeBeliefEstimator(model=model, max_window=5)
        seq, _, _, _ = _sequence()
        ctx = _context(seq.cases[0], chosen="spend")
        est.advance(ctx)
        assert 0.0 <= est.current_belief() <= 1.0
        assert est._history.previous_route == "spend"

    def test_belief_clipped_to_unit_interval(self) -> None:
        # A ridge model whose bias alone pushes the prediction outside
        # [0, 1] must still be clipped by the estimator.
        model = fit_ridge([[0.0] * 55], [5.0])
        est = RidgeBeliefEstimator(model=model, max_window=5)
        seq, _, _, _ = _sequence()
        ctx = _context(seq.cases[0])
        est.advance(ctx)
        assert est.current_belief() == 1.0

    def test_window_bounded_at_max_window(self) -> None:
        model = fit_ridge([[0.0] * 22], [0.5])
        est = RidgeBeliefEstimator(model=model, max_window=2)
        seq, _, _, _ = _sequence()
        for t, case in enumerate(seq.cases[:5]):
            est.advance(_context(case, step=t))
        assert len(est._window) == 2


class TestLookupBeliefEstimator:
    def test_first_call_returns_initial_belief(self) -> None:
        est = LookupBeliefEstimator(beliefs=[0.2, 0.4, 0.6])
        assert est.current_belief() == INITIAL_BELIEF

    def test_advances_through_precomputed_sequence(self) -> None:
        est = LookupBeliefEstimator(beliefs=[0.2, 0.4, 0.6])
        seq, _, _, _ = _sequence()
        ctx = _context(seq.cases[0])
        est.advance(ctx)
        assert est.current_belief() == 0.2
        est.advance(ctx)
        assert est.current_belief() == 0.4
        est.advance(ctx)
        assert est.current_belief() == 0.6

    def test_custom_initial_belief(self) -> None:
        est = LookupBeliefEstimator(beliefs=[0.9], initial_belief=0.1)
        assert est.current_belief() == 0.1


class TestBeliefPricingController:
    def test_initial_lambda_price_matches_oracle(self) -> None:
        oracle = BellmanOracle()
        est = ExactBeliefEstimator(belief=0.5)
        controller = BeliefPricingController(
            oracle=oracle, belief_estimator=est, total_steps=3, initial_budget=4.0
        )
        expected = oracle.marginal_price(3, 4.0, 0.5)
        assert controller.lambda_price["budget"] == pytest.approx(expected)

    def test_explicit_lambda_price_at_construction_is_not_overwritten(self) -> None:
        oracle = BellmanOracle()
        est = ExactBeliefEstimator(belief=0.5)
        controller = BeliefPricingController(
            oracle=oracle,
            belief_estimator=est,
            total_steps=3,
            initial_budget=4.0,
            lambda_price={"budget": 99.0},
        )
        assert controller.lambda_price["budget"] == 99.0

    def test_update_advances_estimator_and_recomputes_price(self) -> None:
        oracle = BellmanOracle()
        seq, _, belief_priors, _ = _sequence()
        est = ExactBeliefEstimator(belief=belief_priors[0])
        controller = BeliefPricingController(
            oracle=oracle, belief_estimator=est, total_steps=len(seq.cases), initial_budget=8.0
        )
        ctx = _context(
            seq.cases[0], step=0, total_steps=len(seq.cases), remaining_before=8.0, remaining_after=7.5
        )
        controller.update(ctx)
        expected_belief = belief_priors[1]
        expected_price = oracle.marginal_price(len(seq.cases) - 1, 7.5, expected_belief)
        assert controller.lambda_price["budget"] == pytest.approx(expected_price)

    def test_runs_end_to_end_through_simulate_policy(self) -> None:
        oracle = BellmanOracle()
        seq, _, belief_priors, _ = _sequence(seed=11)
        est = ExactBeliefEstimator(belief=belief_priors[0])
        controller = BeliefPricingController(
            oracle=oracle,
            belief_estimator=est,
            total_steps=len(seq.cases),
            initial_budget=seq.initial_budget["budget"],
        )
        result, decisions = simulate_policy(seq, pricing_controller=controller)
        assert len(decisions) == len(seq.cases)
        assert result.cumulative_utility >= 0.0


class TestBuildBeliefTrainingPairs:
    def test_produces_one_fewer_example_than_steps(self) -> None:
        oracle = BellmanOracle()
        seq, _, belief_priors, _ = _sequence(seed=5)
        est = ExactBeliefEstimator(belief=belief_priors[0])
        controller = BeliefPricingController(
            oracle=oracle,
            belief_estimator=est,
            total_steps=len(seq.cases),
            initial_budget=seq.initial_budget["budget"],
        )
        _, decisions = simulate_policy(seq, pricing_controller=controller)
        features, targets = build_belief_training_pairs(seq, decisions, belief_priors, max_window=5)
        assert len(features) == len(seq.cases) - 1
        assert len(targets) == len(seq.cases) - 1
        assert targets == belief_priors[1:]

    def test_feature_dimension_matches_flattened_window(self) -> None:
        oracle = BellmanOracle()
        seq, _, belief_priors, _ = _sequence(seed=6)
        est = ExactBeliefEstimator(belief=belief_priors[0])
        controller = BeliefPricingController(
            oracle=oracle,
            belief_estimator=est,
            total_steps=len(seq.cases),
            initial_budget=seq.initial_budget["budget"],
        )
        _, decisions = simulate_policy(seq, pricing_controller=controller)
        max_window = 4
        features, _ = build_belief_training_pairs(seq, decisions, belief_priors, max_window=max_window)
        assert features[0].shape == (max_window * 11,)
