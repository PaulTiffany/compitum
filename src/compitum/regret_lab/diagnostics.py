"""Premature-conservation diagnostics (tranche 4).

A per-step HEURISTIC attribution, not a rigorous decomposition of the
exact end-to-end regret (a joint, path-dependent quantity) -- reported to
distinguish hoarding from genuine scarcity, never as a replacement for the
primary cumulative-regret metric. See
docs/adr/0004-pricing-controller-repair.md.
"""

from __future__ import annotations

from typing import Dict, List

from .environment import DynamicSequence
from .hindsight import HindsightResult
from .pricing import total_available_over_horizon
from .simulator import PolicyDecision


def conservation_depletion_split(
    seq: DynamicSequence,
    decisions: List[PolicyDecision],
    hindsight: HindsightResult,
    scarcity_fraction: float = 0.1,
) -> Dict[str, float]:
    """At each step where the policy's choice yields less utility than the
    hindsight oracle's own choice at that same step, attribute the utility
    gap to "conservation" (resources were NOT genuinely scarce at decision
    time -- any shortfall reflects hoarding, not necessity) or "depletion"
    (at least one resource was below ``scarcity_fraction`` of its total
    available-over-horizon amount -- a genuinely scarce moment). The two
    buckets need not sum exactly to the true end-to-end regret, since
    hindsight's choices are jointly optimal across the whole sequence while
    this attribution scores each step in isolation; the residual is
    reported explicitly as ``unattributed_regret`` rather than hidden.
    """
    total_available = total_available_over_horizon(seq)
    regret_from_conservation = 0.0
    regret_from_depletion = 0.0
    realized_utility = 0.0

    for t, case in enumerate(seq.cases):
        policy_choice = decisions[t].chosen
        hindsight_choice = hindsight.choices[t]
        policy_utility = case.base_utility.get(policy_choice, 0.0)
        hindsight_utility = case.base_utility.get(hindsight_choice, 0.0)
        realized_utility += policy_utility

        gap = hindsight_utility - policy_utility
        if gap <= 0:
            continue
        remaining_before = decisions[t].remaining_before
        scarce = any(
            remaining_before[r] < scarcity_fraction * total_available[r] for r in seq.resource_names
        )
        if scarce:
            regret_from_depletion += gap
        else:
            regret_from_conservation += gap

    total_regret = hindsight.value - realized_utility
    attributed = regret_from_conservation + regret_from_depletion
    return {
        "regret_from_conservation": regret_from_conservation,
        "regret_from_depletion": regret_from_depletion,
        "total_regret": total_regret,
        "unattributed_regret": total_regret - attributed,
    }
