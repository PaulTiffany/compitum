"""Belief-estimator-agnostic pricing controller (tranche 6, Part C).

``BeliefPricingController`` is the single wiring shared by every arm in the
paired pilot: it holds a ``BellmanOracle`` and a pluggable ``BeliefEstimator``,
and exposes exactly the two things ``simulate_policy`` (tranche 3, unchanged)
needs -- a ``lambda_price`` dict and an ``update(context)`` hook. Only the
*source* of the belief estimate varies across arms (exact Bayesian filter,
generic HMM filter, frozen ridge regression, or a frozen FabricPC model via
``LookupBeliefEstimator``); the price derivation
(``BellmanOracle.marginal_price``) and the routing rule (``price_utilities``,
unchanged since tranche 3) are identical for all of them. This isolates
belief-estimation quality as the only thing that can differ between arms,
per docs/adr/0007-belief-state-fabricpc-bellman-pricing.md.

Convention (must match training-data generation in
``experiments/fabricpc/tranche6``): the channel vector describing step
``t`` is built from ``remaining_before`` (budget entering step ``t``),
``steps_left = total_steps - t`` (steps remaining including ``t``), and
``history`` reflecting routes/outcomes through step ``t - 1``. Once step
``t``'s outcome is known, that vector is appended to the rolling window used
to predict ``belief_prior`` for step ``t + 1`` -- never step ``t`` itself,
since every quantity in it (including this step's own opportunity signal)
is legitimately past information by the time step ``t + 1`` is priced.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Protocol, Sequence, Tuple

import numpy as np

from .belief_bellman import BellmanOracle
from .belief_channels import (
    CHANNEL_DIMENSION,
    BeliefChannelHistory,
    advance_belief_history,
    compute_belief_channel_vector,
)
from .belief_hmm_filter import belief_high_from_scalar, hmm_filter_step
from .belief_regime import (
    INITIAL_BELIEF,
    P_OPPORTUNITY,
    REGIME_HIGH,
    REGIME_NORMAL,
    TRANSITION,
    filtered_belief,
    predict_belief,
)
from .environment import DynamicSequence
from .pricing import PricingUpdateContext
from .simulator import PolicyDecision
from .windowed_predictor import RidgeModel, flatten_window, predict_ridge

_TRANSITION_MATRIX = np.array(
    [
        [TRANSITION[REGIME_NORMAL][REGIME_NORMAL], TRANSITION[REGIME_NORMAL][REGIME_HIGH]],
        [TRANSITION[REGIME_HIGH][REGIME_NORMAL], TRANSITION[REGIME_HIGH][REGIME_HIGH]],
    ]
)


def _observed_opportunity(context: PricingUpdateContext) -> bool:
    assert context.case is not None
    return context.case.base_utility.get("opportunity", 0.0) > 0.0


def _hmm_likelihood(observed_opportunity: bool) -> np.ndarray:
    if observed_opportunity:
        return np.array([P_OPPORTUNITY[REGIME_NORMAL], P_OPPORTUNITY[REGIME_HIGH]])
    return np.array([1.0 - P_OPPORTUNITY[REGIME_NORMAL], 1.0 - P_OPPORTUNITY[REGIME_HIGH]])


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


class BeliefEstimator(Protocol):
    """Structural interface every arm's belief source implements.
    ``current_belief()`` returns the estimate for the step about to be
    priced; ``advance(context)`` is called once per step, after that
    step's outcome is known, to produce the estimate for the next step."""

    def current_belief(self) -> float: ...

    def advance(self, context: PricingUpdateContext) -> None: ...


@dataclass
class ExactBeliefEstimator:
    """Arm 3: the environment's own closed-form scalar Bayesian filter --
    the oracle-belief upper bound. Legitimate, not hindsight: at each step
    it consumes only this step's own observed opportunity signal, exactly
    as a real online policy could."""

    belief: float = INITIAL_BELIEF

    def current_belief(self) -> float:
        return self.belief

    def advance(self, context: PricingUpdateContext) -> None:
        observed = _observed_opportunity(context)
        posterior = filtered_belief(self.belief, observed)
        self.belief = predict_belief(posterior)


@dataclass
class HmmBeliefEstimator:
    """Arm 4: the strongest simple structured baseline -- a generic
    matrix-based HMM forward filter given the environment's true
    transition/emission parameters as known model constants (the
    "practitioner has already estimated the regime dynamics" scenario).
    Mathematically equivalent to ``ExactBeliefEstimator`` by construction
    (both compute the exact Bayes-optimal posterior); kept as an
    independently-coded arm per the ADR so Gate B is evaluated honestly
    (see docs/adr/0007's Part B note on this equivalence) rather than
    silently reusing arm 3's implementation."""

    belief_vector: np.ndarray = field(
        default_factory=lambda: belief_high_from_scalar(INITIAL_BELIEF)
    )

    def current_belief(self) -> float:
        return float(self.belief_vector[1])

    def advance(self, context: PricingUpdateContext) -> None:
        observed = _observed_opportunity(context)
        likelihood = _hmm_likelihood(observed)
        _, next_prior = hmm_filter_step(self.belief_vector, _TRANSITION_MATRIX, likelihood)
        self.belief_vector = next_prior


@dataclass
class RidgeBeliefEstimator:
    """Arm 5: "ordinary sequential predictor" -- a plain ridge regression
    over the flattened declared channel window (``windowed_predictor``),
    fit offline on training sequences and frozen at test time. An honest
    scope simplification for a literal small neural network, stated as
    such in the tranche 6 report."""

    model: RidgeModel
    max_window: int
    _belief: float = INITIAL_BELIEF
    _history: BeliefChannelHistory = field(default_factory=BeliefChannelHistory)
    _window: Deque[np.ndarray] = field(default_factory=deque)

    def current_belief(self) -> float:
        return self._belief

    def advance(self, context: PricingUpdateContext) -> None:
        assert context.case is not None and context.chosen is not None
        remaining = context.remaining_before["budget"]
        steps_left = context.total_steps - context.step
        vector = compute_belief_channel_vector(
            remaining, context.case, self._history, steps_left, context.total_steps
        )
        self._window.append(vector)
        while len(self._window) > self.max_window:
            self._window.popleft()
        flattened = flatten_window(list(self._window), self.max_window, CHANNEL_DIMENSION)
        self._belief = _clip01(predict_ridge(self.model, flattened.tolist()))
        self._history = advance_belief_history(self._history, context.chosen, context.case)


@dataclass
class LookupBeliefEstimator:
    """Arms 6/7/8: a precomputed, cached sequence of per-step belief
    estimates (e.g. from a batched FabricPC forward pass run once per
    sequence, or a shuffled permutation of another arm's already-computed
    beliefs). Consumes no live model calls during ``simulate_policy`` --
    satisfies the "observe once, reuse" caching requirement generically
    for any offline-precomputed belief source."""

    beliefs: Sequence[float]
    initial_belief: float = INITIAL_BELIEF
    _index: int = 0

    def current_belief(self) -> float:
        return self.initial_belief if self._index == 0 else self.beliefs[self._index - 1]

    def advance(self, context: PricingUpdateContext) -> None:
        self._index += 1


@dataclass
class BeliefPricingController:
    """Wires any ``BeliefEstimator`` into ``simulate_policy``'s
    ``PricingController`` protocol via the exact Bellman marginal price."""

    oracle: BellmanOracle
    belief_estimator: BeliefEstimator
    total_steps: int
    initial_budget: float
    lambda_price: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lambda_price:
            belief = self.belief_estimator.current_belief()
            price = self.oracle.marginal_price(self.total_steps, self.initial_budget, belief)
            self.lambda_price = {"budget": price}

    def update(self, context: PricingUpdateContext) -> None:
        self.belief_estimator.advance(context)
        remaining_steps = context.total_steps - context.step - 1
        budget = context.remaining_after["budget"]
        belief = self.belief_estimator.current_belief()
        price = self.oracle.marginal_price(remaining_steps, budget, belief)
        self.lambda_price = {"budget": price}


def build_belief_training_pairs(
    seq: DynamicSequence,
    decisions: List[PolicyDecision],
    belief_priors: Sequence[float],
    max_window: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """Builds ``(flattened_window, target_next_belief_prior)`` pairs from
    one completed on-policy rollout. ``decisions`` comes from running
    ``simulate_policy`` with a ``BeliefPricingController`` driven by
    ``ExactBeliefEstimator`` on a TRAINING sequence (the on-policy
    reference used to collect realistic route-choice history for every
    learned estimator; declared explicitly in the tranche 6 report). Uses
    the identical windowing convention as ``RidgeBeliefEstimator``/
    ``LookupBeliefEstimator`` so offline-fit predictors see the same
    inputs at train and test time. One example per step except the last
    (no known next-step prior beyond the sequence horizon)."""
    history = BeliefChannelHistory()
    window: Deque[np.ndarray] = deque()
    features: List[np.ndarray] = []
    targets: List[float] = []
    total_steps = len(seq.cases)
    for t, (case, decision) in enumerate(zip(seq.cases, decisions)):
        remaining = decision.remaining_before["budget"]
        steps_left = total_steps - t
        vector = compute_belief_channel_vector(remaining, case, history, steps_left, total_steps)
        window.append(vector)
        while len(window) > max_window:
            window.popleft()
        if t + 1 < len(belief_priors):
            flattened = flatten_window(list(window), max_window, CHANNEL_DIMENSION)
            features.append(flattened)
            targets.append(belief_priors[t + 1])
        history = advance_belief_history(history, decision.chosen, case)
    return features, targets
