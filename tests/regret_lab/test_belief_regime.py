"""Hidden-regime environment -- exact Bayesian filter hand-checks and
generator structural/determinism checks."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.belief_regime import (
    CONSUMPTION,
    INFEASIBLE_CONSUMPTION,
    MODEL_NAMES,
    P_OPPORTUNITY,
    REGIME_HIGH,
    REGIME_NORMAL,
    UTILITY,
    filtered_belief,
    generate_belief_dataset,
    generate_belief_sequence,
    observation_probability,
    predict_belief,
)


def test_filtered_belief_matches_hand_computed_bayes_update() -> None:
    # P(o=1|HIGH)=0.35, P(o=1|NORMAL)=0.05, prior=0.5
    # posterior(HIGH|o=1) = 0.5*0.35 / (0.5*0.35 + 0.5*0.05) = 0.175/0.2 = 0.875
    posterior = filtered_belief(0.5, True)
    assert posterior == pytest.approx(0.875)


def test_filtered_belief_o_zero_favors_normal() -> None:
    # posterior(HIGH|o=0) = 0.5*0.65 / (0.5*0.65+0.5*0.95) = 0.325/0.8 = 0.40625
    posterior = filtered_belief(0.5, False)
    assert posterior == pytest.approx(0.40625)


def test_filtered_belief_degenerate_prior_stays_put() -> None:
    # A belief of exactly 0 or 1 combined with any observation must not
    # divide by zero, and should remain at the boundary.
    assert filtered_belief(0.0, True) == pytest.approx(0.0)
    assert filtered_belief(1.0, False) == pytest.approx(1.0)


def test_predict_belief_matches_hand_computed_transition() -> None:
    # predict_belief(posterior=1.0) = T[HIGH][HIGH] = 0.6
    assert predict_belief(1.0) == pytest.approx(0.6)
    # predict_belief(posterior=0.0) = T[NORMAL][HIGH] = 0.2
    assert predict_belief(0.0) == pytest.approx(0.2)


def test_observation_probability_matches_hand_computation() -> None:
    # P(o=1|belief=0.5) = 0.5*0.35 + 0.5*0.05 = 0.2
    assert observation_probability(0.5, True) == pytest.approx(0.2)
    assert observation_probability(0.5, False) == pytest.approx(0.8)


def test_observation_probabilities_sum_to_one() -> None:
    for belief in (0.0, 0.1, 0.5, 0.9, 1.0):
        total = observation_probability(belief, True) + observation_probability(belief, False)
        assert total == pytest.approx(1.0)


def test_generate_belief_sequence_deterministic() -> None:
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    seq_a, regimes_a, priors_a, posteriors_a = generate_belief_sequence(rng_a, "s", steps=10)
    seq_b, regimes_b, priors_b, posteriors_b = generate_belief_sequence(rng_b, "s", steps=10)
    assert regimes_a == regimes_b
    assert priors_a == priors_b
    assert posteriors_a == posteriors_b
    for ca, cb in zip(seq_a.cases, seq_b.cases):
        assert ca.to_dict() == cb.to_dict()


def test_generate_belief_sequence_structure() -> None:
    rng = np.random.default_rng(1)
    seq, regimes, priors, posteriors = generate_belief_sequence(rng, "s", steps=10)
    assert len(seq.cases) == 10
    assert len(regimes) == 10
    assert len(priors) == 10
    assert len(posteriors) == 10
    assert seq.model_names == MODEL_NAMES
    assert all(r in (REGIME_NORMAL, REGIME_HIGH) for r in regimes)
    assert all(0.0 <= p <= 1.0 for p in priors)
    assert all(0.0 <= p <= 1.0 for p in posteriors)


def test_generate_belief_sequence_opportunity_feasibility_matches_realized_draw() -> None:
    rng = np.random.default_rng(2)
    seq, _, _, _ = generate_belief_sequence(rng, "s", steps=20)
    for case in seq.cases:
        opp_cost = case.realized_consumption["opportunity"]["budget"]
        opp_utility = case.base_utility["opportunity"]
        available = opp_cost < INFEASIBLE_CONSUMPTION
        if available:
            assert opp_cost == CONSUMPTION["opportunity"]
            assert opp_utility == UTILITY["opportunity"]
        else:
            assert opp_utility == 0.0


def test_generate_belief_sequence_conserve_and_spend_always_feasible_economics() -> None:
    rng = np.random.default_rng(3)
    seq, _, _, _ = generate_belief_sequence(rng, "s", steps=5)
    for case in seq.cases:
        assert case.realized_consumption["conserve"]["budget"] == CONSUMPTION["conserve"]
        assert case.realized_consumption["spend"]["budget"] == CONSUMPTION["spend"]
        assert case.base_utility["conserve"] == UTILITY["conserve"]
        assert case.base_utility["spend"] == UTILITY["spend"]


def test_generate_belief_dataset_size_and_uniqueness() -> None:
    dataset = generate_belief_dataset(seed=42, n_sequences=10)
    assert len(dataset) == 10
    ids = {seq.sequence_id for seq, *_ in dataset}
    assert len(ids) == 10


def test_generate_belief_dataset_deterministic() -> None:
    a = generate_belief_dataset(seed=7, n_sequences=5)
    b = generate_belief_dataset(seed=7, n_sequences=5)
    for (seq_a, regimes_a, priors_a, posteriors_a), (
        seq_b,
        regimes_b,
        priors_b,
        posteriors_b,
    ) in zip(a, b):
        assert regimes_a == regimes_b
        assert priors_a == priors_b
        for ca, cb in zip(seq_a.cases, seq_b.cases):
            assert ca.to_dict() == cb.to_dict()


def test_belief_evolves_toward_high_after_repeated_high_observations() -> None:
    # A synthetic run of consecutive "opportunity available" observations
    # should push belief upward over time (regardless of any specific
    # generated sequence) -- a basic sanity check on the filter direction.
    belief = 0.1
    for _ in range(5):
        posterior = filtered_belief(belief, True)
        belief = predict_belief(posterior)
    assert belief > 0.1


def test_belief_evolves_toward_normal_after_repeated_low_observations() -> None:
    belief = 0.9
    for _ in range(5):
        posterior = filtered_belief(belief, False)
        belief = predict_belief(posterior)
    assert belief < 0.9


def test_p_opportunity_and_transition_are_valid_probabilities() -> None:
    assert 0.0 <= P_OPPORTUNITY[REGIME_NORMAL] <= 1.0
    assert 0.0 <= P_OPPORTUNITY[REGIME_HIGH] <= 1.0
    assert P_OPPORTUNITY[REGIME_HIGH] > P_OPPORTUNITY[REGIME_NORMAL]
