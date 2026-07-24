"""Generic N-state HMM forward filter (tranche 6, arm 4: "simple Bayesian/
HMM filter"). Deliberately implemented independently of
``belief_regime.py``'s hardcoded 2-regime scalar formulas -- using generic
matrix/vector operations, not the environment's own closed-form shortcuts
-- so it serves both as the required "strongest simple structured
baseline" and as a from-scratch cross-check that the environment's exact
filter is correct (see ``tests/regret_lab/test_belief_hmm_filter.py``, which
verifies both give the same belief trajectory).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def hmm_filter_step(
    belief_prior: np.ndarray, transition: np.ndarray, likelihood: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """One generic HMM forward-filter step.

    ``belief_prior``: prior distribution over states, shape ``(n,)``.
    ``transition``: ``transition[i, j] = P(next=j | current=i)``, shape
    ``(n, n)``.
    ``likelihood``: ``likelihood[i] = P(observation | state=i)``, shape
    ``(n,)``.

    Returns ``(posterior, next_prior)``: the Bayes-updated belief after
    this observation, and that posterior projected forward one transition
    step (the next step's prior).
    """
    unnormalized = belief_prior * likelihood
    total = unnormalized.sum()
    posterior = unnormalized / total
    next_prior = transition.T @ posterior
    return posterior, next_prior


def belief_high_from_scalar(belief_prior_high: float) -> np.ndarray:
    """Converts this environment's scalar P(regime=HIGH) representation
    into a full 2-state distribution ``[P(NORMAL), P(HIGH)]`` for use with
    the generic filter above."""
    return np.array([1.0 - belief_prior_high, belief_prior_high])
