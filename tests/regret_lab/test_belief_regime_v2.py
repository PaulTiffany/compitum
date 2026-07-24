"""Belief-sensitive environment (tranche 7) -- regime-dependent
"opportunity" payoff, hidden from every policy exactly like regime
itself; only the belief-weighted expectation is ever knowable."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.belief_regime_v2 import (
    REGIME_HIGH,
    REGIME_NORMAL,
    expected_opportunity_utility,
    generate_belief_dataset_v2,
    generate_belief_sequence_v2,
)


class TestExpectedOpportunityUtility:
    def test_posterior_zero_gives_normal_value(self) -> None:
        assert expected_opportunity_utility(0.0, u_normal=1.0, u_high=8.0) == pytest.approx(1.0)

    def test_posterior_one_gives_high_value(self) -> None:
        assert expected_opportunity_utility(1.0, u_normal=1.0, u_high=8.0) == pytest.approx(8.0)

    def test_posterior_half_gives_average(self) -> None:
        assert expected_opportunity_utility(0.5, u_normal=1.0, u_high=8.0) == pytest.approx(4.5)

    def test_monotonic_increasing_in_posterior(self) -> None:
        low = expected_opportunity_utility(0.2, u_normal=1.0, u_high=8.0)
        high = expected_opportunity_utility(0.8, u_normal=1.0, u_high=8.0)
        assert high > low


class TestGenerateBeliefSequenceV2:
    def test_returns_same_shape_as_v1(self) -> None:
        rng = np.random.default_rng(0)
        seq, true_regimes, belief_priors, belief_posteriors = generate_belief_sequence_v2(
            rng, "s0", initial_budget=8.0
        )
        assert len(seq.cases) == len(true_regimes) == len(belief_priors) == len(belief_posteriors)
        assert seq.initial_budget == {"budget": 8.0}
        assert seq.model_names == ("conserve", "spend", "opportunity")

    def test_realized_opportunity_utility_matches_true_regime(self) -> None:
        rng = np.random.default_rng(3)
        seq, true_regimes, _, _ = generate_belief_sequence_v2(
            rng, "s3", initial_budget=8.0, u_normal=1.0, u_high=8.0
        )
        for regime, case in zip(true_regimes, seq.cases):
            available = case.base_utility["opportunity"] > 0.0
            if available:
                expected = 8.0 if regime == REGIME_HIGH else 1.0
                assert case.base_utility["opportunity"] == pytest.approx(expected)
            else:
                assert case.base_utility["opportunity"] == 0.0

    def test_opportunity_infeasible_when_unavailable(self) -> None:
        rng = np.random.default_rng(5)
        seq, _, _, _ = generate_belief_sequence_v2(rng, "s5", initial_budget=8.0)
        for case in seq.cases:
            if case.base_utility["opportunity"] == 0.0:
                assert case.realized_consumption["opportunity"]["budget"] > 1e5

    def test_custom_u_normal_u_high_applied(self) -> None:
        rng = np.random.default_rng(7)
        seq, true_regimes, _, _ = generate_belief_sequence_v2(
            rng, "s7", initial_budget=8.0, u_normal=2.0, u_high=10.0
        )
        seen_values = {
            case.base_utility["opportunity"]
            for case in seq.cases
            if case.base_utility["opportunity"] > 0.0
        }
        assert seen_values <= {2.0, 10.0}

    def test_true_regimes_are_valid_regime_values(self) -> None:
        rng = np.random.default_rng(9)
        _, true_regimes, _, _ = generate_belief_sequence_v2(rng, "s9", initial_budget=8.0)
        assert set(true_regimes) <= {REGIME_NORMAL, REGIME_HIGH}


class TestGenerateBeliefDatasetV2:
    def test_produces_declared_number_of_sequences(self) -> None:
        data = generate_belief_dataset_v2(seed=1, n_sequences=5, initial_budget=8.0)
        assert len(data) == 5

    def test_sequence_ids_use_declared_prefix(self) -> None:
        data = generate_belief_dataset_v2(seed=1, n_sequences=2, id_prefix="custom", initial_budget=8.0)
        assert data[0][0].sequence_id == "custom-0000"
        assert data[1][0].sequence_id == "custom-0001"

    def test_deterministic_given_same_seed(self) -> None:
        data1 = generate_belief_dataset_v2(seed=42, n_sequences=3, initial_budget=8.0)
        data2 = generate_belief_dataset_v2(seed=42, n_sequences=3, initial_budget=8.0)
        for (seq1, r1, p1, po1), (seq2, r2, p2, po2) in zip(data1, data2):
            assert [c.base_utility for c in seq1.cases] == [c.base_utility for c in seq2.cases]
            assert r1 == r2
