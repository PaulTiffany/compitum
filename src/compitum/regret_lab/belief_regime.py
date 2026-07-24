"""Hidden-regime stochastic environment for tranche 6 (economically correct
price oracle). Deliberately small and narrow per the "cost and runtime
discipline" directive: one resource, two hidden regimes, one fixed finite
horizon, no scenario sweep.

Generative model: a 2-state Markov chain (NORMAL, HIGH) governs, each step,
the probability that a high-value "opportunity" action is available. The
policy never observes the regime directly -- only whether the opportunity
was available this step, which is itself the sole stochastic observation
used for exact Bayesian regime filtering. Ground-truth regimes and exact
belief trajectories are returned SEPARATELY from the ``DynamicSequence``
(never placed in any field a policy reads), for training targets and
diagnostics only. See docs/adr/0007-belief-state-fabricpc-bellman-pricing.md.

Reuses ``DynamicCase``/``DynamicSequence`` (and therefore ``simulate_policy``,
``PacingController``, etc.) unchanged: the three declared models
(``conserve``, ``spend``, ``opportunity``) are always present per case,
exactly like ``scarcity_scenarios.py``, with ``opportunity`` priced to be
unconditionally infeasible on steps where it did not become available.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .environment import DynamicCase, DynamicSequence

REGIME_NORMAL = 0
REGIME_HIGH = 1
REGIME_NAMES = ("normal", "high")

# TRANSITION[from_regime][to_regime]
TRANSITION: Dict[int, Dict[int, float]] = {
    REGIME_NORMAL: {REGIME_NORMAL: 0.8, REGIME_HIGH: 0.2},
    REGIME_HIGH: {REGIME_NORMAL: 0.4, REGIME_HIGH: 0.6},
}
P_OPPORTUNITY: Dict[int, float] = {REGIME_NORMAL: 0.05, REGIME_HIGH: 0.35}

MODEL_NAMES: Tuple[str, str, str] = ("conserve", "spend", "opportunity")
UTILITY: Dict[str, float] = {"conserve": 1.0, "spend": 2.0, "opportunity": 8.0}
CONSUMPTION: Dict[str, float] = {"conserve": 1.0, "spend": 2.0, "opportunity": 4.0}
INFEASIBLE_CONSUMPTION = 1.0e6

GRID_UNIT = 0.5  # the natural discrete resource unit (gcd of all consumption/replenishment amounts)
REPLENISHMENT = 0.5
STEPS = 10
INITIAL_BUDGET = 8.0
INITIAL_BELIEF = 0.5


def _observation_likelihoods(observed_opportunity: bool) -> Tuple[float, float]:
    """(P(observation | HIGH), P(observation | NORMAL)) for the given
    observed value."""
    if observed_opportunity:
        return P_OPPORTUNITY[REGIME_HIGH], P_OPPORTUNITY[REGIME_NORMAL]
    return 1.0 - P_OPPORTUNITY[REGIME_HIGH], 1.0 - P_OPPORTUNITY[REGIME_NORMAL]


def filtered_belief(belief_prior: float, observed_opportunity: bool) -> float:
    """Exact Bayesian posterior P(regime=HIGH) after observing this step's
    opportunity-availability signal, given the prior belief. The
    denominator is always strictly positive here since both
    ``P_OPPORTUNITY`` values are strictly in (0, 1) by construction -- no
    degenerate-prior guard needed."""
    p_high, p_normal = _observation_likelihoods(observed_opportunity)
    numerator_high = belief_prior * p_high
    numerator_normal = (1.0 - belief_prior) * p_normal
    return numerator_high / (numerator_high + numerator_normal)


def predict_belief(belief_posterior: float) -> float:
    """Belief projected forward one regime-transition step, becoming the
    next step's prior."""
    return (
        belief_posterior * TRANSITION[REGIME_HIGH][REGIME_HIGH]
        + (1.0 - belief_posterior) * TRANSITION[REGIME_NORMAL][REGIME_HIGH]
    )


def observation_probability(belief_prior: float, observed_opportunity: bool) -> float:
    """P(this step's observation | current prior belief), marginalizing
    over the (unknown) current regime."""
    p_high, p_normal = _observation_likelihoods(observed_opportunity)
    return belief_prior * p_high + (1.0 - belief_prior) * p_normal


def _case_for_step(step: int, opportunity_available: bool, rng: np.random.Generator) -> DynamicCase:
    utility = {
        "conserve": UTILITY["conserve"],
        "spend": UTILITY["spend"],
        "opportunity": UTILITY["opportunity"] if opportunity_available else 0.0,
    }
    consumption = {
        "conserve": {"budget": CONSUMPTION["conserve"]},
        "spend": {"budget": CONSUMPTION["spend"]},
        "opportunity": {
            "budget": (
                CONSUMPTION["opportunity"] if opportunity_available else INFEASIBLE_CONSUMPTION
            )
        },
    }
    return DynamicCase(
        step=step,
        base_utility=utility,
        expected_consumption=consumption,
        realized_consumption=consumption,
        revelation_delay=0,
        replenishment={"budget": REPLENISHMENT},
    )


def generate_belief_sequence(
    rng: np.random.Generator,
    sequence_id: str,
    steps: int = STEPS,
    initial_budget: float = INITIAL_BUDGET,
    initial_belief: float = INITIAL_BELIEF,
) -> Tuple[DynamicSequence, List[int], List[float], List[float]]:
    """Returns ``(sequence, true_regimes, belief_priors, belief_posteriors)``.
    The regime/belief lists are ground truth for training targets and
    diagnostics ONLY -- never placed in any field a policy can read."""
    true_regimes: List[int] = []
    belief_priors: List[float] = []
    belief_posteriors: List[float] = []
    cases: List[DynamicCase] = []

    regime = REGIME_HIGH if rng.random() < initial_belief else REGIME_NORMAL
    belief_prior = initial_belief

    for t in range(steps):
        true_regimes.append(regime)
        belief_priors.append(belief_prior)
        opportunity_available = bool(rng.random() < P_OPPORTUNITY[regime])
        posterior = filtered_belief(belief_prior, opportunity_available)
        belief_posteriors.append(posterior)

        cases.append(_case_for_step(t, opportunity_available, rng))

        next_probs = TRANSITION[regime]
        regime = REGIME_HIGH if rng.random() < next_probs[REGIME_HIGH] else REGIME_NORMAL
        belief_prior = predict_belief(posterior)

    sequence = DynamicSequence(
        sequence_id=sequence_id,
        scenario="belief_regime",
        resource_names=("budget",),
        model_names=MODEL_NAMES,
        initial_budget={"budget": initial_budget},
        cases=cases,
    )
    return sequence, true_regimes, belief_priors, belief_posteriors


def generate_belief_dataset(
    seed: int,
    n_sequences: int,
    steps: int = STEPS,
    initial_budget: float = INITIAL_BUDGET,
    initial_belief: float = INITIAL_BELIEF,
    id_prefix: str = "belief",
) -> List[Tuple[DynamicSequence, List[int], List[float], List[float]]]:
    out = []
    for index in range(n_sequences):
        rng = np.random.default_rng((seed, index))
        sequence_id = f"{id_prefix}-{index:04d}"
        out.append(
            generate_belief_sequence(rng, sequence_id, steps, initial_budget, initial_belief)
        )
    return out
