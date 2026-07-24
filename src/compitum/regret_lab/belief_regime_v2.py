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

This module changes only the payoff process, per the authorizing brief:
the regime now controls how much "opportunity" is actually worth when it
appears (``U_NORMAL_OPPORTUNITY`` in the NORMAL regime, ``U_HIGH_OPPORTUNITY``
in HIGH), not merely whether it appears. The realized payoff is drawn
according to the TRUE hidden regime and recorded in
``DynamicCase.base_utility["opportunity"]`` for ground-truth utility/regret
accounting -- but the true regime remains hidden from every policy, exactly
as before; the ONLY additional information any policy gets is its own
belief, from which it must form an expectation
(``(1 - q) * U_NORMAL_OPPORTUNITY + q * U_HIGH_OPPORTUNITY``) to decide
whether taking "opportunity" now is worthwhile. See
docs/adr/0009-belief-sensitive-shadow-charge-validation.md.

Regime dynamics (``TRANSITION``, ``P_OPPORTUNITY``, hidden-regime
filtering formulas) are reused unchanged from ``belief_regime.py``; only
``U_NORMAL_OPPORTUNITY``, ``U_HIGH_OPPORTUNITY``, and ``INITIAL_BUDGET``
are tunable, via Gate 0's tiny preregistered development grid
(``experiments/fabricpc/tranche7/run_gate0_identifiability.py``), chosen
solely by belief-boundary occupancy -- never by FabricPC/ridge/backprop
performance.
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
    filtered_belief,
    observation_probability,
    predict_belief,
)
from .environment import DynamicCase, DynamicSequence

__all__ = [
    "CONSUMPTION",
    "GRID_UNIT",
    "INFEASIBLE_CONSUMPTION",
    "INITIAL_BELIEF",
    "MODEL_NAMES",
    "P_OPPORTUNITY",
    "REGIME_HIGH",
    "REGIME_NAMES",
    "REGIME_NORMAL",
    "REPLENISHMENT",
    "STEPS",
    "TRANSITION",
    "UTILITY",
    "U_HIGH_OPPORTUNITY_DEFAULT",
    "U_NORMAL_OPPORTUNITY_DEFAULT",
    "expected_opportunity_utility",
    "filtered_belief",
    "generate_belief_sequence_v2",
    "generate_belief_dataset_v2",
    "observation_probability",
    "predict_belief",
]

# Defaults only; Gate 0 (experiments/fabricpc/tranche7/run_gate0_identifiability.py)
# selects the frozen configuration actually used by the tranche-7 pilot
# from a tiny declared grid around these values, chosen purely by
# belief-boundary occupancy.
U_NORMAL_OPPORTUNITY_DEFAULT = 1.0
U_HIGH_OPPORTUNITY_DEFAULT = 8.0


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
) -> Tuple[DynamicSequence, List[int], List[float], List[float]]:
    """Returns ``(sequence, true_regimes, belief_priors, belief_posteriors)``,
    exactly like ``belief_regime.generate_belief_sequence`` -- the regime
    -transition and observation dynamics are identical; only the realized
    "opportunity" payoff, recorded in ``base_utility["opportunity"]`` for
    ground-truth accounting only, now depends on the hidden regime."""
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

        cases.append(_case_for_step_v2(t, opportunity_available, regime, u_normal, u_high))

        next_probs = TRANSITION[regime]
        regime = REGIME_HIGH if rng.random() < next_probs[REGIME_HIGH] else REGIME_NORMAL
        belief_prior = predict_belief(posterior)

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
    id_prefix: str = "belief-v2",
) -> List[Tuple[DynamicSequence, List[int], List[float], List[float]]]:
    out = []
    for index in range(n_sequences):
        rng = np.random.default_rng((seed, index))
        sequence_id = f"{id_prefix}-{index:04d}"
        out.append(
            generate_belief_sequence_v2(
                rng, sequence_id, steps, initial_budget, initial_belief, u_normal, u_high
            )
        )
    return out
