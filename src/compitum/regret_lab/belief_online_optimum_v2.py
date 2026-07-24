"""Exact Bayes-optimal online policy for the belief-sensitive environment
(tranche 7).

``belief_online_optimum.run_online_optimal_policy`` was reused unchanged
for tranche 7's FIRST Gate 0 pass (payoff/budget tuning only), since it
never scores an action itself -- it only calls
``oracle.best_action_given_observation(...)``. But it also tracks its
OWN belief internally, via ``belief_regime.filtered_belief``/``predict_belief``,
which hardcode tranche 6's FIXED observation/transition parameters. Once
Gate 0's development grid needed to tune those parameters too (see
``belief_regime_v2.py``'s module docstring for why payoff/budget tuning
alone was not enough), reusing the original unchanged would have made
this reference policy's own belief silently diverge from the actual
environment being evaluated -- an incorrect "exact online optimum" is
worse than a merely-suboptimal one, since every regret number in the
pilot is measured against it. This is a minimal, parameterized sibling,
not a new algorithm: identical structure, only the two belief-update
calls change.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from .belief_regime_v2 import (
    P_OPPORTUNITY_HIGH_DEFAULT,
    P_OPPORTUNITY_NORMAL_DEFAULT,
    TRANSITION_HIGH_TO_HIGH_DEFAULT,
    TRANSITION_NORMAL_TO_HIGH_DEFAULT,
    filtered_belief_v2,
    predict_belief_v2,
)
from .environment import DynamicSequence
from .hindsight import HindsightResult
from .metrics import PolicyRunResult
from .simulator import PolicyDecision


def run_online_optimal_policy_v2(
    seq: DynamicSequence,
    oracle: Any,
    initial_belief: float,
    p_opportunity_normal: float = P_OPPORTUNITY_NORMAL_DEFAULT,
    p_opportunity_high: float = P_OPPORTUNITY_HIGH_DEFAULT,
    transition_normal_to_high: float = TRANSITION_NORMAL_TO_HIGH_DEFAULT,
    transition_high_to_high: float = TRANSITION_HIGH_TO_HIGH_DEFAULT,
) -> Tuple[PolicyRunResult, List[PolicyDecision]]:
    """``violation_count``/``violation_magnitude`` and
    ``depleted_budget_events`` remain always zero for the same provable
    reasons as ``belief_online_optimum.run_online_optimal_policy``
    (unaffected by the payoff change: feasibility and
    consumption/replenishment arithmetic are untouched). Unlike that
    function, ``high_value_rejections`` is tracked live, not hardoded to
    zero: since "opportunity"'s true payoff now depends on the hidden
    regime, even the exact-belief, Bayes-optimal decision can turn out,
    in hindsight, to have rejected a strictly-higher-true-utility option
    -- an honest consequence of genuine residual uncertainty, not a
    policy defect."""
    total_steps = len(seq.cases)
    remaining: Dict[str, float] = dict(seq.initial_budget)
    belief_prior = initial_belief
    cumulative_utility = 0.0
    choices: List[str] = []
    decisions: List[PolicyDecision] = []
    deferral_count = 0
    avoidable_deferral_count = 0
    high_value_rejection_count = 0
    total_consumption: Dict[str, float] = {r: 0.0 for r in seq.resource_names}

    for t, case in enumerate(seq.cases):
        start = time.perf_counter()
        remaining_before = dict(remaining)
        observed_opportunity = case.base_utility.get("opportunity", 0.0) > 0.0
        chosen, _ = oracle.best_action_given_observation(
            total_steps - t, remaining["budget"], belief_prior, observed_opportunity
        )
        feasible_now = [
            m
            for m in seq.model_names
            if all(
                remaining[r] - case.realized_consumption[m][r] >= -1e-9 for r in seq.resource_names
            )
        ]
        if chosen != "defer":
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

        posterior = filtered_belief_v2(
            belief_prior, observed_opportunity, p_opportunity_normal, p_opportunity_high
        )
        belief_prior = predict_belief_v2(
            posterior, transition_normal_to_high, transition_high_to_high
        )

        latency = time.perf_counter() - start
        decisions.append(
            PolicyDecision(
                step=t,
                chosen=chosen,
                feasible_models=feasible_now,
                priced_utility=dict(case.base_utility),
                violation_magnitude_so_far=0.0,
                latency_seconds=latency,
                remaining_before=remaining_before,
                lambda_price_before={},
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
    return result, decisions


def online_optimum_as_hindsight_result_v2(
    seq: DynamicSequence,
    oracle: Any,
    initial_belief: float,
    p_opportunity_normal: float = P_OPPORTUNITY_NORMAL_DEFAULT,
    p_opportunity_high: float = P_OPPORTUNITY_HIGH_DEFAULT,
    transition_normal_to_high: float = TRANSITION_NORMAL_TO_HIGH_DEFAULT,
    transition_high_to_high: float = TRANSITION_HIGH_TO_HIGH_DEFAULT,
) -> HindsightResult:
    """Packages the exact online optimum's achieved utility/choices as a
    ``HindsightResult`` so ``regret_metrics``/``paired_regret_deltas``/
    ``bootstrap_ci`` (all unchanged) can be pointed at it as an
    alternative reference optimum to true hindsight."""
    result, _ = run_online_optimal_policy_v2(
        seq,
        oracle,
        initial_belief,
        p_opportunity_normal,
        p_opportunity_high,
        transition_normal_to_high,
        transition_high_to_high,
    )
    return HindsightResult(
        value=result.cumulative_utility,
        choices=result.choices,
        exact=True,
        optimality_gap=0.0,
        state_count=oracle.state_count(),
    )
