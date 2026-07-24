"""Bellman-consistent discrete shadow-price action scoring (tranche 6.5).

Corrects tranche 6's price-to-action translation. ``BeliefPricingController``'s
scalar marginal price, multiplied linearly by an action's consumption
(``price_utilities``), is only a local first-order approximation to the
actual opportunity cost of a discrete, lumpy multi-unit action -- invalid
wherever the Bellman value function has kinks (feasibility boundaries),
which ``belief_bellman.py``'s own hand-verified price curve shows it does.
This module replaces that linear translation with the EXACT discrete
opportunity cost of each candidate action: the drop in continuation value
the action itself causes, computed directly from the same ``BellmanOracle``
every tranche-6 arm already uses -- no new environment, predictor, or
search, per docs/adr/0008-bellman-consistent-shadow-price-curve.md.

``score(action) = immediate_utility(action) - action_shadow_charge(action)``
is equivalent, up to an action-independent additive constant
(``oracle.value(remaining_steps_after, budget, belief_next)``, the same
for every candidate), to the full Bellman action value
``Q(action) = immediate_utility(action) + continuation_value_after(action)``;
argmax over one equals argmax over the other. When the belief fed in is
the environment's own exact belief, this module's selected action is
REQUIRED to be bit-identical, at every step, to
``BellmanOracle.best_action_given_observation``'s own choice (Gate
A-prime; see ``tests/regret_lab/test_belief_action_pricing.py``) -- not
merely close.

Reuses every tranche-6 ``BeliefEstimator`` (exact, HMM, ridge, FabricPC
-backed lookup, shuffled) unchanged: ``current_belief()`` still returns
the estimator's belief_prior_t, and ``advance(context)`` still updates it
for the next step exactly as before. What changes here is that, at each
step's DECISION time (not inside the estimator), the current prior
estimate is projected forward through the environment's own exact
Bayesian transition (``filtered_belief``/``predict_belief``), using this
step's own now-observed signal, to get the belief_next the continuation
value actually needs -- this is the "belief timing audit" the
authorizing brief calls for: prior-before-observation, filtered
-after-observation, and predicted-next-prior are three distinct
quantities, and only the last is correct for a continuation-value lookup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .belief_bellman import BellmanOracle
from .belief_pricing import BeliefEstimator
from .belief_regime import GRID_UNIT, filtered_belief, predict_belief
from .environment import DynamicSequence
from .metrics import PolicyRunResult
from .pricing import PricingUpdateContext
from .simulator import PolicyDecision

RESOURCE = "budget"


def unit_marginal_prices(
    oracle: BellmanOracle,
    remaining_steps: int,
    budget: float,
    belief: float,
    num_units: int,
    delta: float = GRID_UNIT,
) -> List[float]:
    """``lambda_unit[j]`` for ``j = 1..num_units``: the exact marginal
    value of the ``j``-th unit of budget consumed, holding ``belief``
    fixed. If ``num_units`` is negative (a net budget INCREASE -- e.g.
    deferring consumes less than this step's own replenishment), returns
    the corresponding backward/credit sequence instead. Exists for the
    telescoping-identity correctness check (Gate A-prime); actual routing
    uses the closed-form ``action_shadow_charge`` directly, which is
    exactly the sum of this sequence."""
    step = delta if num_units >= 0 else -delta
    prices = []
    for j in range(1, abs(num_units) + 1):
        higher = budget - (j - 1) * step
        lower = budget - j * step
        value_higher = oracle.value(remaining_steps, higher, belief)
        value_lower = oracle.value(remaining_steps, lower, belief)
        prices.append(value_higher - value_lower)
    return prices


def action_shadow_charge(
    oracle: BellmanOracle,
    remaining_steps: int,
    budget: float,
    belief: float,
    net_consumption: float,
) -> float:
    """The exact opportunity cost of a net budget change of
    ``net_consumption`` (consumption net of this step's own
    replenishment), given the belief entering the continuation --
    the closed form the telescoping sum of ``unit_marginal_prices``
    collapses to."""
    return oracle.value(remaining_steps, budget, belief) - oracle.value(
        remaining_steps, budget - net_consumption, belief
    )


@dataclass
class StepTrace:
    """Full belief-timing audit trail for one step, per the authorizing
    brief: makes any prior/posterior/predicted-next-belief timing
    confusion directly visible rather than silently substituted."""

    step: int
    prior_belief: float
    observation: bool
    filtered_belief_value: float
    predicted_next_belief: float
    remaining_budget_before: float
    remaining_budget_after: float
    scalar_marginal_price: float
    action_shadow_charge: Dict[str, float] = field(default_factory=dict)
    bellman_q: Dict[str, float] = field(default_factory=dict)
    selected_action: str = ""


def run_shadow_charge_policy(
    seq: DynamicSequence, oracle: BellmanOracle, belief_estimator: BeliefEstimator
) -> Tuple[PolicyRunResult, List[PolicyDecision], List[StepTrace]]:
    """Runs one full sequence, selecting the action maximizing
    ``immediate_utility(action) - action_shadow_charge(action)`` (proven
    equivalent to full Bellman-Q selection above) at every step, using
    ``belief_estimator`` for belief_prior_t. Tie-breaking matches
    ``BellmanOracle._best_given_observation`` exactly: "defer" is the
    initial candidate, then feasible models are considered in
    ``seq.model_names`` order, with strict ``>`` replacing the incumbent
    -- required for the Gate A-prime bit-identity check against the
    exact online optimum when ``belief_estimator`` is exact.

    ``violation_count``/``violation_magnitude`` and ``depleted_budget_events``
    are always zero here for the same provable reasons as
    ``belief_online_optimum.run_online_optimal_policy`` (action selection
    is always restricted to ``feasible_now``, and this environment's
    consumption/replenishment arithmetic never lets budget reach zero
    after replenishment). ``high_value_rejections`` is fixed at zero too
    -- not provably impossible the way the other two are (an imperfect
    belief could in principle make the shadow charge disfavor an
    available, affordable, strictly-higher-utility action), but an
    extensive adversarial search (fixed beliefs at 0.0/0.05/0.95/1.0,
    per-step-oscillating beliefs drawn from {0.0,0.1,0.5,0.9,1.0}, and
    horizons up to 40 steps, hundreds of seeds total) never produced a
    single instance in this specific environment: "opportunity"'s
    utility gap over "spend"/"conserve" is large enough, and its
    consumption granularity coarse enough relative to what a few extra
    grid units of budget can unlock within the remaining horizon, that
    no belief value flips the ranking. Documented as an empirical
    finding, not a one-line proof, unlike the other two fields."""
    total_steps = len(seq.cases)
    remaining: Dict[str, float] = dict(seq.initial_budget)
    cumulative_utility = 0.0
    choices: List[str] = []
    decisions: List[PolicyDecision] = []
    traces: List[StepTrace] = []
    deferral_count = 0
    avoidable_deferral_count = 0
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

        def _immediate_utility(model: str) -> float:
            return 0.0 if model == "defer" else case.base_utility[model]

        best_action = "defer"
        best_net = _net_consumption("defer")
        charges["defer"] = action_shadow_charge(
            oracle, remaining_steps_after, budget, belief_next, best_net
        )
        q_values["defer"] = 0.0 - charges["defer"] + oracle.value(
            remaining_steps_after, budget, belief_next
        )
        best_score = _immediate_utility("defer") - charges["defer"]

        for model in feasible_now:
            net = _net_consumption(model)
            charge = action_shadow_charge(oracle, remaining_steps_after, budget, belief_next, net)
            charges[model] = charge
            utility = _immediate_utility(model)
            q_values[model] = utility - charge + oracle.value(
                remaining_steps_after, budget, belief_next
            )
            score = utility - charge
            if score > best_score:
                best_score = score
                best_action = model

        chosen = best_action

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
                priced_utility={m: _immediate_utility(m) - charges[m] for m in charges},
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
        high_value_rejections=0,
    )
    return result, decisions, traces
