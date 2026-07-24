"""Exact online Bellman value function for the belief-sensitive
environment (tranche 7).

A new class, not a modification of the frozen ``BellmanOracle`` (tranche
6/6.5's own 100%-covered, Gate-A-prime-proven implementation is reused
completely unchanged and untouched by this file) -- per the authorizing
brief's "modify only the payoff process," this reuses the identical
recursive value-function algorithm, memoization strategy, and discrete
-shadow-charge-compatible method signatures
(``value``/``marginal_price``/``best_action_given_observation``/``state_count``),
changing only how the recursion values the "opportunity" branch: instead
of a single fixed constant, it is the belief-weighted expectation
``(1 - posterior) * u_normal + posterior * u_high`` -- the same
computation ``belief_regime_v2.expected_opportunity_utility`` performs,
inlined here since the oracle already has the posterior in hand from its
own recursion.

Because ``belief_action_pricing.action_shadow_charge``/``unit_marginal_prices``
only ever call ``oracle.value(...)`` -- structurally, not by concrete
type -- they work with this oracle completely unchanged; see
``belief_action_pricing_v2.py`` for the one place the immediate-utility
formula for "opportunity" also needs to change, for the analogous reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .belief_regime import CONSUMPTION, GRID_UNIT, MODEL_NAMES, REPLENISHMENT, UTILITY
from .belief_regime_v2 import (
    P_OPPORTUNITY_HIGH_DEFAULT,
    P_OPPORTUNITY_NORMAL_DEFAULT,
    TRANSITION_HIGH_TO_HIGH_DEFAULT,
    TRANSITION_NORMAL_TO_HIGH_DEFAULT,
    U_HIGH_OPPORTUNITY_DEFAULT,
    U_NORMAL_OPPORTUNITY_DEFAULT,
    filtered_belief_v2,
    observation_probability_v2,
    predict_belief_v2,
)

_BELIEF_ROUND_DIGITS = 9


class BellmanStateBudgetExceeded(Exception):
    pass


@dataclass
class BeliefSensitiveBellmanOracle:
    u_normal_opportunity: float = U_NORMAL_OPPORTUNITY_DEFAULT
    u_high_opportunity: float = U_HIGH_OPPORTUNITY_DEFAULT
    p_opportunity_normal: float = P_OPPORTUNITY_NORMAL_DEFAULT
    p_opportunity_high: float = P_OPPORTUNITY_HIGH_DEFAULT
    transition_normal_to_high: float = TRANSITION_NORMAL_TO_HIGH_DEFAULT
    transition_high_to_high: float = TRANSITION_HIGH_TO_HIGH_DEFAULT
    max_states: int = 2_000_000
    _value_memo: Dict[Tuple[int, float, float], float] = field(default_factory=dict, repr=False)

    def _feasible_models(self, budget: float, observed_opportunity: bool) -> Tuple[str, ...]:
        return tuple(
            m
            for m in MODEL_NAMES
            if (m != "opportunity" or observed_opportunity) and CONSUMPTION[m] <= budget + 1e-9
        )

    def _opportunity_utility(self, posterior_belief_high: float) -> float:
        return (
            1.0 - posterior_belief_high
        ) * self.u_normal_opportunity + posterior_belief_high * self.u_high_opportunity

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

        total = 0.0
        for observed in (False, True):
            p_o = observation_probability_v2(
                belief_prior, observed, self.p_opportunity_normal, self.p_opportunity_high
            )
            _, best_value = self._best_given_observation(
                remaining_steps, budget_key, belief_prior, observed
            )
            total += p_o * best_value
        self._value_memo[key] = total
        return total

    def _best_given_observation(
        self, remaining_steps: int, budget: float, belief_prior: float, observed_opportunity: bool
    ) -> Tuple[str, float]:
        posterior = filtered_belief_v2(
            belief_prior, observed_opportunity, self.p_opportunity_normal, self.p_opportunity_high
        )
        belief_next = predict_belief_v2(
            posterior, self.transition_normal_to_high, self.transition_high_to_high
        )

        best_action = "defer"
        best_value = self.value(remaining_steps - 1, budget + REPLENISHMENT, belief_next)

        for model in self._feasible_models(budget, observed_opportunity):
            if model == "opportunity":
                utility = self._opportunity_utility(posterior)
            else:
                utility = UTILITY[model]
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
