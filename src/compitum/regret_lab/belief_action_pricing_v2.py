"""Bellman-consistent discrete shadow-price action scoring for the
belief-sensitive environment (tranche 7).

Reuses ``action_shadow_charge``/``unit_marginal_prices``/``StepTrace``
from ``belief_action_pricing.py`` completely unchanged -- the discrete
-shadow-charge FORMULA never changes, per the authorizing brief ("no
more pricing changes"). What genuinely must change here, and could not
be reused unchanged, is how a candidate action's IMMEDIATE utility is
computed for scoring: ``belief_regime_v2.DynamicCase.base_utility["opportunity"]``
stores the TRUE realized payoff (drawn from the hidden regime, for
ground-truth cumulative-utility/regret accounting only) -- using it
directly for the SCORE, the way ``run_shadow_charge_policy`` does, would
let the policy see the true regime-dependent value before deciding,
which is exactly the leakage this whole program's design discipline
forbids. This module instead scores "opportunity" using the
belief-weighted EXPECTATION
(``belief_regime_v2.expected_opportunity_utility``, using the estimator's
own posterior -- imperfect for weak belief sources, by design), while
still crediting the true realized payoff to cumulative utility once
chosen, exactly like every other tranche's honest expectation-vs-reality
split.

``belief_online_optimum.run_online_optimal_policy`` needed NO equivalent
change and is reused completely unchanged for this environment too: it
never computes a per-action score itself, only calling
``oracle.best_action_given_observation(...)``, which (via
``BeliefSensitiveBellmanOracle``) already performs the correct
belief-weighted computation internally.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from .belief_action_pricing import StepTrace, action_shadow_charge
from .belief_regime import filtered_belief, predict_belief
from .belief_regime_v2 import expected_opportunity_utility
from .environment import DynamicSequence
from .metrics import PolicyRunResult
from .pricing import PricingUpdateContext
from .simulator import PolicyDecision

RESOURCE = "budget"


def run_shadow_charge_policy_v2(
    seq: DynamicSequence,
    oracle: Any,
    belief_estimator: Any,
    u_normal: float,
    u_high: float,
) -> Tuple[PolicyRunResult, List[PolicyDecision], List[StepTrace]]:
    """Identical routing logic to ``belief_action_pricing.run_shadow_charge_policy``
    (same tie-breaking, same reliance on ``action_shadow_charge`` for the
    continuation-value term) except that a candidate "opportunity"'s
    scoring utility is ``expected_opportunity_utility(posterior, u_normal, u_high)``
    -- the estimator's own posterior, not the case's true realized value."""
    total_steps = len(seq.cases)
    remaining: Dict[str, float] = dict(seq.initial_budget)
    cumulative_utility = 0.0
    choices: List[str] = []
    decisions: List[PolicyDecision] = []
    traces: List[StepTrace] = []
    deferral_count = 0
    avoidable_deferral_count = 0
    high_value_rejection_count = 0
    total_consumption: Dict[str, float] = {r: 0.0 for r in seq.resource_names}

    for t, case in enumerate(seq.cases):
        start = time.perf_counter()
        remaining_before = dict(remaining)
        budget = remaining[RESOURCE]
        observed_opportunity = case.base_utility.get("opportunity", 0.0) > 0.0

        prior = belief_estimator.current_belief()
        posterior = filtered_belief(prior, observed_opportunity)
        belief_next = predict_belief(posterior)
        remaining_steps_after = total_steps - t - 1
        scalar_price = oracle.marginal_price(total_steps - t, budget, prior)

        feasible_now = [
            m
            for m in seq.model_names
            if all(
                remaining[r] - case.realized_consumption[m][r] >= -1e-9 for r in seq.resource_names
            )
        ]

        charges: Dict[str, float] = {}
        q_values: Dict[str, float] = {}

        def _net_consumption(model: str) -> float:
            if model == "defer":
                return -case.replenishment[RESOURCE]
            return case.realized_consumption[model][RESOURCE] - case.replenishment[RESOURCE]

        def _scoring_utility(model: str) -> float:
            if model == "defer":
                return 0.0
            if model == "opportunity":
                return expected_opportunity_utility(posterior, u_normal, u_high)
            return case.base_utility[model]

        best_action = "defer"
        best_net = _net_consumption("defer")
        charges["defer"] = action_shadow_charge(
            oracle, remaining_steps_after, budget, belief_next, best_net
        )
        q_values["defer"] = 0.0 - charges["defer"] + oracle.value(
            remaining_steps_after, budget, belief_next
        )
        best_score = _scoring_utility("defer") - charges["defer"]

        for model in feasible_now:
            net = _net_consumption(model)
            charge = action_shadow_charge(oracle, remaining_steps_after, budget, belief_next, net)
            charges[model] = charge
            utility = _scoring_utility(model)
            q_values[model] = utility - charge + oracle.value(
                remaining_steps_after, budget, belief_next
            )
            score = utility - charge
            if score > best_score:
                best_score = score
                best_action = model

        chosen = best_action

        if chosen != "defer":
            # Hindsight-informed diagnostic only (never fed back into
            # scoring): did the decision -- necessarily made from belief,
            # not the true regime -- turn out to reject a feasible
            # option with strictly higher TRUE realized utility? Unlike
            # tranche 6/6.5's environment, this can genuinely happen here
            # once "opportunity"'s true payoff depends on the hidden
            # regime, so it is tracked live rather than assumed zero.
            best_by_true_utility = max(seq.model_names, key=lambda m: case.base_utility[m])
            if chosen != best_by_true_utility and best_by_true_utility in feasible_now:
                high_value_rejection_count += 1
            cumulative_utility += case.base_utility[chosen]
            for r in seq.resource_names:
                remaining[r] -= case.realized_consumption[chosen][r]
                total_consumption[r] += case.realized_consumption[chosen][r]
        else:
            deferral_count += 1
            if feasible_now:
                avoidable_deferral_count += 1

        for r in seq.resource_names:
            remaining[r] += case.replenishment[r]

        belief_estimator.advance(
            PricingUpdateContext(
                resource_names=seq.resource_names,
                reservation={},
                remaining_before=remaining_before,
                remaining_after=dict(remaining),
                step=t,
                total_steps=total_steps,
                case=case,
                chosen=chosen,
            )
        )

        latency = time.perf_counter() - start
        decisions.append(
            PolicyDecision(
                step=t,
                chosen=chosen,
                feasible_models=feasible_now,
                priced_utility={m: _scoring_utility(m) - charges[m] for m in charges},
                violation_magnitude_so_far=0.0,
                latency_seconds=latency,
                remaining_before=remaining_before,
                lambda_price_before={RESOURCE: scalar_price},
            )
        )
        traces.append(
            StepTrace(
                step=t,
                prior_belief=prior,
                observation=observed_opportunity,
                filtered_belief_value=posterior,
                predicted_next_belief=belief_next,
                remaining_budget_before=remaining_before[RESOURCE],
                remaining_budget_after=remaining[RESOURCE],
                scalar_marginal_price=scalar_price,
                action_shadow_charge=charges,
                bellman_q=q_values,
                selected_action=chosen,
            )
        )
        choices.append(chosen)

    route_switch_count = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i - 1])
    result = PolicyRunResult(
        sequence_id=seq.sequence_id,
        cumulative_utility=cumulative_utility,
        choices=choices,
        violation_count=0,
        violation_magnitude=0.0,
        deferral_count=deferral_count,
        avoidable_deferral_count=avoidable_deferral_count,
        route_switch_count=route_switch_count,
        depleted_budget_events=0,
        total_consumption=total_consumption,
        decision_latencies=[d.latency_seconds for d in decisions],
        terminal_remaining=dict(remaining),
        high_value_rejections=high_value_rejection_count,
    )
    return result, decisions, traces
