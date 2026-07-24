"""Generic HMM forward filter -- cross-validated against belief_regime.py's
independent, hardcoded 2-regime scalar formulas."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.belief_hmm_filter import belief_high_from_scalar, hmm_filter_step
from compitum.regret_lab.belief_regime import (
    P_OPPORTUNITY,
    REGIME_HIGH,
    REGIME_NORMAL,
    TRANSITION,
    filtered_belief,
    predict_belief,
)

_TRANSITION_MATRIX = np.array(
    [
        [TRANSITION[REGIME_NORMAL][REGIME_NORMAL], TRANSITION[REGIME_NORMAL][REGIME_HIGH]],
        [TRANSITION[REGIME_HIGH][REGIME_NORMAL], TRANSITION[REGIME_HIGH][REGIME_HIGH]],
    ]
)


def _likelihood(observed_opportunity: bool) -> np.ndarray:
    if observed_opportunity:
        return np.array([P_OPPORTUNITY[REGIME_NORMAL], P_OPPORTUNITY[REGIME_HIGH]])
    return np.array([1.0 - P_OPPORTUNITY[REGIME_NORMAL], 1.0 - P_OPPORTUNITY[REGIME_HIGH]])


@pytest.mark.parametrize("belief_high", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("observed", [True, False])
def test_generic_filter_matches_exact_scalar_filter(belief_high: float, observed: bool) -> None:
    prior = belief_high_from_scalar(belief_high)
    posterior, next_prior = hmm_filter_step(prior, _TRANSITION_MATRIX, _likelihood(observed))

    expected_posterior_high = filtered_belief(belief_high, observed)
    expected_next_prior_high = predict_belief(expected_posterior_high)

    assert posterior[1] == pytest.approx(expected_posterior_high, abs=1e-9)
    assert next_prior[1] == pytest.approx(expected_next_prior_high, abs=1e-9)


def test_generic_filter_matches_across_a_multi_step_trajectory() -> None:
    rng = np.random.default_rng(0)
    belief_high = 0.5
    prior_vec = belief_high_from_scalar(belief_high)
    for _ in range(20):
        observed = bool(rng.random() < 0.2)
        posterior_vec, prior_vec = hmm_filter_step(
            prior_vec, _TRANSITION_MATRIX, _likelihood(observed)
        )

        posterior_scalar = filtered_belief(belief_high, observed)
        belief_high = predict_belief(posterior_scalar)

        assert posterior_vec[1] == pytest.approx(posterior_scalar, abs=1e-9)
        assert prior_vec[1] == pytest.approx(belief_high, abs=1e-9)


def test_posterior_sums_to_one() -> None:
    prior = belief_high_from_scalar(0.3)
    posterior, next_prior = hmm_filter_step(prior, _TRANSITION_MATRIX, _likelihood(True))
    assert posterior.sum() == pytest.approx(1.0)
    assert next_prior.sum() == pytest.approx(1.0)


def test_belief_high_from_scalar_conversion() -> None:
    vec = belief_high_from_scalar(0.3)
    assert vec[0] == pytest.approx(0.7)
    assert vec[1] == pytest.approx(0.3)
