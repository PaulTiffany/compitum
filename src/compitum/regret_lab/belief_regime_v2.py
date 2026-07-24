"""Belief-sensitive hidden-regime environment (tranche 7).

Tranche 6.5 proved the shadow-charge price-to-action translation exactly
correct and recovered the full economic gap over pacing -- but also
found, directly, that no reachable action in that environment's
parameterization ever depended on belief: "opportunity"'s payoff was a
single fixed constant regardless of the hidden regime, so belief only
ever affected a Bellman *value* used for regret accounting, never an
*argmax over actions*. Every belief source (exact, learned, even
shuffled) therefore tied at zero regret -- benchmark unidentifiability,
not evidence about FabricPC.

This module changes the payoff process, per the authorizing brief: the
regime now controls how much "opportunity" is actually worth when it
appears (``U_NORMAL_OPPORTUNITY`` in the NORMAL regime,
``U_HIGH_OPPORTUNITY`` in HIGH), not merely whether it appears. The
realized payoff is drawn according to the TRUE hidden regime and
recorded in ``DynamicCase.base_utility["opportunity"]`` for ground-truth
utility/regret accounting -- but the true regime remains hidden from
every policy, exactly as before; the ONLY additional information any
policy gets is its own belief, from which it must form an expectation
(``(1 - q) * U_NORMAL_OPPORTUNITY + q * U_HIGH_OPPORTUNITY``) to decide
whether taking "opportunity" now is worthwhile. See
docs/adr/0009-belief-sensitive-shadow-charge-validation.md.

Gate 0's first pass (four payoff/budget-only combinations) found genuine
belief-dependent decision boundaries but an exact tie between the exact
-belief policy and simple belief-blind controls (fixed prior, shuffled):
directly diagnosed as an observation-informativeness artifact --
``P_OPPORTUNITY[HIGH]=0.35`` vs ``P_OPPORTUNITY[NORMAL]=0.05`` (tranche
6/6.5's fixed values) makes a SINGLE observation of "opportunity
available" so informative on its own that a naive single-step read
(``filtered_belief(0.5, True) = 0.875``) already lands on the same side
of the decision boundary as several steps of accumulated exact belief,
regardless of history. Per the brief's own suggested tunable dimensions
("observation noise", "transition persistence"), this module therefore
also parameterizes ``P_OPPORTUNITY``/``TRANSITION`` (via
``filtered_belief_v2``/``predict_belief_v2``/``observation_probability_v2``,
independent reimplementations of ``belief_regime.py``'s fixed-parameter
formulas, defaulting to those exact fixed values for
byte-for-byte-compatible behavior when not overridden). Gate 0's
development grid (``experiments/fabricpc/tranche7/run_gate0_identifiability.py``)
tunes these alongside ``U_NORMAL_OPPORTUNITY``/``U_HIGH_OPPORTUNITY``/
``INITIAL_BUDGET``, chosen solely by belief-boundary occupancy and the
exact-belief policy's margin over belief-blind controls -- never by
FabricPC/ridge/backprop performance.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .belief_regime import (
    CONSUMPTION,
    GRID_UNIT,
    INFEASIBLE_CONSUMPTION,
    INITIAL_BELIEF,
    MODEL_NAMES,
    P_OPPORTUNITY,
    REGIME_HIGH,
    REGIME_NAMES,
    REGIME_NORMAL,
    REPLENISHMENT,
    STEPS,
    TRANSITION,
    UTILITY,
)
from .environment import DynamicCase, DynamicSequence

__all__ = [
    "CONSUMPTION",
    "GRID_UNIT",
    "INFEASIBLE_CONSUMPTION",
    "INITIAL_BELIEF",
    "MODEL_NAMES",
    "P_OPPORTUNITY",
    "P_OPPORTUNITY_HIGH_DEFAULT",
    "P_OPPORTUNITY_NORMAL_DEFAULT",
    "REGIME_HIGH",
    "REGIME_NAMES",
    "REGIME_NORMAL",
    "REPLENISHMENT",
    "STEPS",
    "TRANSITION",
    "TRANSITION_HIGH_TO_HIGH_DEFAULT",
    "TRANSITION_NORMAL_TO_HIGH_DEFAULT",
    "UTILITY",
    "U_HIGH_OPPORTUNITY_DEFAULT",
    "U_NORMAL_OPPORTUNITY_DEFAULT",
    "expected_opportunity_utility",
    "filtered_belief_v2",
    "generate_belief_sequence_v2",
    "generate_belief_dataset_v2",
    "observation_probability_v2",
    "predict_belief_v2",
]

# Defaults only; Gate 0 (experiments/fabricpc/tranche7/run_gate0_identifiability.py)
# selects the frozen configuration actually used by the tranche-7 pilot
# from a tiny declared grid around these values, chosen purely by
# belief-boundary occupancy and the exact-belief-vs-belief-blind margin.
U_NORMAL_OPPORTUNITY_DEFAULT = 1.0
U_HIGH_OPPORTUNITY_DEFAULT = 8.0
# Matches belief_regime.py's fixed P_OPPORTUNITY/TRANSITION exactly, so
# calling any function below without overriding these reproduces
# tranche 6/6.5's own regime dynamics byte-for-byte.
P_OPPORTUNITY_NORMAL_DEFAULT = P_OPPORTUNITY[REGIME_NORMAL]
P_OPPORTUNITY_HIGH_DEFAULT = P_OPPORTUNITY[REGIME_HIGH]
TRANSITION_NORMAL_TO_HIGH_DEFAULT = TRANSITION[REGIME_NORMAL][REGIME_HIGH]
TRANSITION_HIGH_TO_HIGH_DEFAULT = TRANSITION[REGIME_HIGH][REGIME_HIGH]


def expected_opportunity_utility(
    posterior_belief_high: float,
    u_normal: float = U_NORMAL_OPPORTUNITY_DEFAULT,
    u_high: float = U_HIGH_OPPORTUNITY_DEFAULT,
) -> float:
    """The belief-weighted expected payoff of taking "opportunity" right
    now, given the posterior (not prior) belief that the regime is HIGH
    -- the only thing any policy, including the exact-belief oracle, ever
    gets to use to value it, since the true regime stays hidden even
    when "opportunity" is known to be available."""
    return (1.0 - posterior_belief_high) * u_normal + posterior_belief_high * u_high


def _observation_likelihoods_v2(
    observed_opportunity: bool, p_opportunity_normal: float, p_opportunity_high: float
) -> Tuple[float, float]:
    """(P(observation | HIGH), P(observation | NORMAL))."""
    if observed_opportunity:
        return p_opportunity_high, p_opportunity_normal
    return 1.0 - p_opportunity_high, 1.0 - p_opportunity_normal


def filtered_belief_v2(
    belief_prior: float,
    observed_opportunity: bool,
    p_opportunity_normal: float = P_OPPORTUNITY_NORMAL_DEFAULT,
    p_opportunity_high: float = P_OPPORTUNITY_HIGH_DEFAULT,
) -> float:
    """Independent reimplementation of ``belief_regime.filtered_belief``,
    parameterized by observation informativeness -- identical formula,
    identical output when called with the default parameters."""
    p_high, p_normal = _observation_likelihoods_v2(
        observed_opportunity, p_opportunity_normal, p_opportunity_high
    )
    numerator_high = belief_prior * p_high
    numerator_normal = (1.0 - belief_prior) * p_normal
    return numerator_high / (numerator_high + numerator_normal)


def predict_belief_v2(
    belief_posterior: float,
    transition_normal_to_high: float = TRANSITION_NORMAL_TO_HIGH_DEFAULT,
    transition_high_to_high: float = TRANSITION_HIGH_TO_HIGH_DEFAULT,
) -> float:
    """Independent reimplementation of ``belief_regime.predict_belief``,
    parameterized by regime persistence."""
    return (
        belief_posterior * transition_high_to_high
        + (1.0 - belief_posterior) * transition_normal_to_high
    )


def observation_probability_v2(
    belief_prior: float,
    observed_opportunity: bool,
    p_opportunity_normal: float = P_OPPORTUNITY_NORMAL_DEFAULT,
    p_opportunity_high: float = P_OPPORTUNITY_HIGH_DEFAULT,
) -> float:
    p_high, p_normal = _observation_likelihoods_v2(
        observed_opportunity, p_opportunity_normal, p_opportunity_high
    )
    return belief_prior * p_high + (1.0 - belief_prior) * p_normal


def _case_for_step_v2(
    step: int,
    opportunity_available: bool,
    regime: int,
    u_normal: float,
    u_high: float,
) -> DynamicCase:
    realized_opportunity_utility = (
        (u_high if regime == REGIME_HIGH else u_normal) if opportunity_available else 0.0
    )
    utility = {
        "conserve": UTILITY["conserve"],
        "spend": UTILITY["spend"],
        "opportunity": realized_opportunity_utility,
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


def generate_belief_sequence_v2(
    rng: np.random.Generator,
    sequence_id: str,
    steps: int = STEPS,
    initial_budget: float = 8.0,
    initial_belief: float = INITIAL_BELIEF,
    u_normal: float = U_NORMAL_OPPORTUNITY_DEFAULT,
    u_high: float = U_HIGH_OPPORTUNITY_DEFAULT,
    p_opportunity_normal: float = P_OPPORTUNITY_NORMAL_DEFAULT,
    p_opportunity_high: float = P_OPPORTUNITY_HIGH_DEFAULT,
    transition_normal_to_high: float = TRANSITION_NORMAL_TO_HIGH_DEFAULT,
    transition_high_to_high: float = TRANSITION_HIGH_TO_HIGH_DEFAULT,
) -> Tuple[DynamicSequence, List[int], List[float], List[float]]:
    """Returns ``(sequence, true_regimes, belief_priors, belief_posteriors)``,
    the same shape as ``belief_regime.generate_belief_sequence``. Regime
    -transition and observation-noise parameters default to tranche 6's
    exact fixed values but are overridable for Gate 0's development grid;
    only the realized "opportunity" payoff, recorded in
    ``base_utility["opportunity"]`` for ground-truth accounting only, is
    always regime-dependent here."""
    true_regimes: List[int] = []
    belief_priors: List[float] = []
    belief_posteriors: List[float] = []
    cases: List[DynamicCase] = []

    regime = REGIME_HIGH if rng.random() < initial_belief else REGIME_NORMAL
    belief_prior = initial_belief

    for t in range(steps):
        true_regimes.append(regime)
        belief_priors.append(belief_prior)
        p_opportunity_this_regime = (
            p_opportunity_high if regime == REGIME_HIGH else p_opportunity_normal
        )
        opportunity_available = bool(rng.random() < p_opportunity_this_regime)
        posterior = filtered_belief_v2(
            belief_prior, opportunity_available, p_opportunity_normal, p_opportunity_high
        )
        belief_posteriors.append(posterior)

        cases.append(_case_for_step_v2(t, opportunity_available, regime, u_normal, u_high))

        next_high_prob = (
            transition_high_to_high if regime == REGIME_HIGH else transition_normal_to_high
        )
        regime = REGIME_HIGH if rng.random() < next_high_prob else REGIME_NORMAL
        belief_prior = predict_belief_v2(
            posterior, transition_normal_to_high, transition_high_to_high
        )

    sequence = DynamicSequence(
        sequence_id=sequence_id,
        scenario="belief_regime_v2",
        resource_names=("budget",),
        model_names=MODEL_NAMES,
        initial_budget={"budget": initial_budget},
        cases=cases,
    )
    return sequence, true_regimes, belief_priors, belief_posteriors


def generate_belief_dataset_v2(
    seed: int,
    n_sequences: int,
    steps: int = STEPS,
    initial_budget: float = 8.0,
    initial_belief: float = INITIAL_BELIEF,
    u_normal: float = U_NORMAL_OPPORTUNITY_DEFAULT,
    u_high: float = U_HIGH_OPPORTUNITY_DEFAULT,
    p_opportunity_normal: float = P_OPPORTUNITY_NORMAL_DEFAULT,
    p_opportunity_high: float = P_OPPORTUNITY_HIGH_DEFAULT,
    transition_normal_to_high: float = TRANSITION_NORMAL_TO_HIGH_DEFAULT,
    transition_high_to_high: float = TRANSITION_HIGH_TO_HIGH_DEFAULT,
    id_prefix: str = "belief-v2",
) -> List[Tuple[DynamicSequence, List[int], List[float], List[float]]]:
    out = []
    for index in range(n_sequences):
        rng = np.random.default_rng((seed, index))
        sequence_id = f"{id_prefix}-{index:04d}"
        out.append(
            generate_belief_sequence_v2(
                rng,
                sequence_id,
                steps,
                initial_budget,
                initial_belief,
                u_normal,
                u_high,
                p_opportunity_normal,
                p_opportunity_high,
                transition_normal_to_high,
                transition_high_to_high,
            )
        )
    return out
