"""Exact Bayes-optimal online policy (tranche 6, Part A principal
comparator).

Directly executes ``BellmanOracle.best_action_given_observation``'s chosen
action at every step -- the true online (non-hindsight) optimum, given
only information available up to and including this step's own
observation. Distinguished from arm 3 (``ExactBeliefEstimator`` +
``BeliefPricingController``'s greedy ``price_utilities`` routing), which
is a *price-based approximation* to this same belief state, not the
literal per-step DP-optimal action -- this module is the un-approximated
ceiling for any ONLINE policy. ``compute_hindsight_optimum`` (tranche 3,
unchanged) remains separate and strictly stronger (it knows the full
future realization in advance; this policy never does), per
docs/adr/0007-belief-state-fabricpc-bellman-pricing.md.

Result packaged as a ``HindsightResult`` (tranche 3's dataclass, reused
unchanged) purely so the existing, frozen ``regret_metrics``/
``paired_regret_deltas``/``bootstrap_ci`` machinery can be pointed at
either comparator interchangeably -- this policy is not a hindsight
optimum and the reuse implies no such claim.

Assumes ``revelation_delay == 0`` and ``expected_consumption ==
realized_consumption`` for every case, exactly as ``belief_regime.py``
always generates. A declared simplification specific to this
environment, not a general-purpose simulator (see ``simulator.py`` for
the general reservation/true-up ledger every other tranche's environment
needs).
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from .belief_bellman import BellmanOracle
from .belief_regime import filtered_belief, predict_belief
from .environment import DynamicSequence
from .hindsight import HindsightResult
from .metrics import PolicyRunResult
from .simulator import PolicyDecision


def run_online_optimal_policy(
    seq: DynamicSequence, oracle: BellmanOracle, initial_belief: float
) -> Tuple[PolicyRunResult, List[PolicyDecision]]:
    """``violation_count``/``violation_magnitude``, ``depleted_budget_events``,
    and ``high_value_rejections`` are always zero for this policy on any
    sequence ``belief_regime.py`` generates -- not approximated as zero,
    provably so given this environment's own parameters, so no defensive
    tracking code is kept for them (matches the project's standing
    preference to remove, not merely leave untested, genuinely
    unreachable branches):

    * No violation is possible: ``oracle.best_action_given_observation``
      only ever returns an action from ``_feasible_models`` (or "defer"),
      and ``_feasible_models`` already requires
      ``CONSUMPTION[m] <= budget``, so realized consumption never exceeds
      available budget.
    * No depletion event is possible: every consumption/replenishment
      amount is an exact multiple of ``GRID_UNIT`` (0.5), the smallest
      feasible nonzero consumption is 1.0 (2 units), and replenishment is
      0.5 (1 unit) every step -- so after replenishment, remaining budget
      in GRID_UNIT-units is always >= 1 (>= 0.5 actual), never <= 0,
      regardless of which feasible action (including "defer") was taken.
    * No high-value rejection is possible: every "opportunity" case has
      the same fixed utility (8.0, strictly dominating "spend" and
      "conserve"), so there is never a reason to prefer a lower-utility
      action while a higher-utility one is both available and
      affordable -- unlike environments with route-specific utility that
      varies in value, not just availability.
    """
    total_steps = len(seq.cases)
    remaining: Dict[str, float] = dict(seq.initial_budget)
    belief_prior = initial_belief
    cumulative_utility = 0.0
    choices: List[str] = []
    decisions: List[PolicyDecision] = []
    deferral_count = 0
    avoidable_deferral_count = 0
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

        posterior = filtered_belief(belief_prior, observed_opportunity)
        belief_prior = predict_belief(posterior)

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
        high_value_rejections=0,
    )
    return result, decisions


def online_optimum_as_hindsight_result(
    seq: DynamicSequence, oracle: BellmanOracle, initial_belief: float
) -> HindsightResult:
    """Packages the exact online optimum's achieved utility/choices as a
    ``HindsightResult`` so ``regret_metrics``/``paired_regret_deltas``/
    ``bootstrap_ci`` (all unchanged) can be pointed at it as an
    alternative reference optimum to true hindsight -- see module
    docstring for why this is not a hindsight claim."""
    result, _ = run_online_optimal_policy(seq, oracle, initial_belief)
    return HindsightResult(
        value=result.cumulative_utility,
        choices=result.choices,
        exact=True,
        optimality_gap=0.0,
        state_count=oracle.state_count(),
    )
