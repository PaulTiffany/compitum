"""Exact online (belief-state) Bellman value function and marginal shadow
price for tranche 6's hidden-regime environment.

This is the economically grounded price the whole tranche is organized
around: the expected loss in optimal future utility caused by having one
less unit of resource, conditional on currently available information
(the belief state) -- not "the lambda needed to reproduce one oracle
action" (tranche 5's construction, which this tranche does not reuse).

Exact, not learned or estimated: belief is a sufficient statistic for this
POMDP (standard result), and because actions never affect the hidden
regime or its observation, the belief trajectory is independent of the
policy's own choices -- the value recursion needs no approximation beyond
floating-point exactness (all consumption/replenishment/budget values are
exact multiples of ``GRID_UNIT``, so no drift accumulates). Memoized on
``(remaining_steps, budget, belief)``; since the reachable belief set from
a fixed initial belief is a finite tree of depth ``steps``, this state
space is bounded and small for the horizons this tranche uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .belief_regime import (
    CONSUMPTION,
    GRID_UNIT,
    MODEL_NAMES,
    REPLENISHMENT,
    UTILITY,
    filtered_belief,
    observation_probability,
    predict_belief,
)

_BELIEF_ROUND_DIGITS = 9


class BellmanStateBudgetExceeded(Exception):
    pass


@dataclass
class BellmanOracle:
    max_states: int = 2_000_000
    _value_memo: Dict[Tuple[int, float, float], float] = field(default_factory=dict, repr=False)

    def _feasible_models(self, budget: float, observed_opportunity: bool) -> Tuple[str, ...]:
        return tuple(
            m
            for m in MODEL_NAMES
            if (m != "opportunity" or observed_opportunity) and CONSUMPTION[m] <= budget + 1e-9
        )

    def value(self, remaining_steps: int, budget: float, belief_prior: float) -> float:
        """Expected value achievable over the remaining horizon, BEFORE
        this step's opportunity observation is known -- marginalizes over
        it using ``belief_prior``."""
        if remaining_steps <= 0:
            return 0.0
        belief_key = round(belief_prior, _BELIEF_ROUND_DIGITS)
        budget_key = round(budget / GRID_UNIT) * GRID_UNIT
        key = (remaining_steps, budget_key, belief_key)
        if key in self._value_memo:
            return self._value_memo[key]
        if len(self._value_memo) >= self.max_states:
            raise BellmanStateBudgetExceeded(
                f"Bellman oracle exceeded declared state cap ({self.max_states})"
            )

        # Both observation branches always have strictly positive
        # probability (P_OPPORTUNITY is strictly in (0, 1) for both
        # regimes), so no zero-probability skip is needed here.
        total = 0.0
        for observed in (False, True):
            p_o = observation_probability(belief_prior, observed)
            _, best_value = self._best_given_observation(
                remaining_steps, budget_key, belief_prior, observed
            )
            total += p_o * best_value
        self._value_memo[key] = total
        return total

    def _best_given_observation(
        self, remaining_steps: int, budget: float, belief_prior: float, observed_opportunity: bool
    ) -> Tuple[str, float]:
        posterior = filtered_belief(belief_prior, observed_opportunity)
        belief_next = predict_belief(posterior)

        # "defer" is just a fourth always-feasible pseudo-model: zero
        # utility, zero consumption, but it still receives replenishment
        # like every other choice.
        best_action = "defer"
        best_value = self.value(remaining_steps - 1, budget + REPLENISHMENT, belief_next)

        for model in self._feasible_models(budget, observed_opportunity):
            utility = UTILITY[model] if (model != "opportunity" or observed_opportunity) else 0.0
            next_budget = budget - CONSUMPTION[model] + REPLENISHMENT
            candidate = utility + self.value(remaining_steps - 1, next_budget, belief_next)
            if candidate > best_value:
                best_value = candidate
                best_action = model
        return best_action, best_value

    def marginal_price(
        self, remaining_steps: int, budget: float, belief_prior: float, delta: float = GRID_UNIT
    ) -> float:
        higher = self.value(remaining_steps, budget, belief_prior)
        lower = self.value(remaining_steps, max(budget - delta, 0.0), belief_prior)
        return (higher - lower) / delta

    def best_action_given_observation(
        self,
        remaining_steps: int,
        budget: float,
        belief_prior: float,
        observed_opportunity: bool,
    ) -> Tuple[str, float]:
        """The Bayes-optimal action given today's ALREADY-OBSERVED
        opportunity signal (not marginalized) -- this is what an online
        policy actually chooses, once it knows whether the opportunity
        exists this step."""
        return self._best_given_observation(
            remaining_steps, budget, belief_prior, observed_opportunity
        )

    def state_count(self) -> int:
        return len(self._value_memo)
