"""Regret and feasibility metrics (tranche 3).

Cumulative constrained regret is the primary metric; violations are always
reported as a separate, visible number, never folded into the regret scalar
as a penalty, per docs/adr/0003-dynamic-constraint-regret.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from .hindsight import HindsightResult


@dataclass
class PolicyRunResult:
    sequence_id: str
    cumulative_utility: float
    choices: List[str]
    violation_count: int
    violation_magnitude: float
    deferral_count: int
    avoidable_deferral_count: int
    route_switch_count: int
    depleted_budget_events: int
    total_consumption: Dict[str, float]
    decision_latencies: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "cumulative_utility": self.cumulative_utility,
            "choices": list(self.choices),
            "violation_count": self.violation_count,
            "violation_magnitude": self.violation_magnitude,
            "deferral_count": self.deferral_count,
            "avoidable_deferral_count": self.avoidable_deferral_count,
            "route_switch_count": self.route_switch_count,
            "depleted_budget_events": self.depleted_budget_events,
            "total_consumption": dict(self.total_consumption),
            "decision_latencies": list(self.decision_latencies),
        }


def regret_metrics(
    policy_results: Sequence[PolicyRunResult],
    hindsight_results: Dict[str, HindsightResult],
) -> Dict[str, float]:
    """Aggregate regret and feasibility metrics across a set of paired
    sequences. ``per_sequence_regret`` is exposed for paired-uncertainty
    analysis (e.g. bootstrap CIs over paired differences between arms)."""
    if not policy_results:
        return {
            "mean_regret": float("nan"),
            "median_regret": float("nan"),
            "tail_regret_p95": float("nan"),
            "total_violation_count": 0.0,
            "total_violation_magnitude": 0.0,
            "total_deferrals": 0.0,
            "total_avoidable_deferrals": 0.0,
            "mean_route_switch_rate": float("nan"),
            "total_depleted_budget_events": 0.0,
            "utility_per_resource_unit": float("nan"),
            "n_sequences": 0.0,
        }

    regrets = np.array(
        [hindsight_results[pr.sequence_id].value - pr.cumulative_utility for pr in policy_results]
    )
    total_utility = sum(pr.cumulative_utility for pr in policy_results)
    total_consumption = sum(sum(pr.total_consumption.values()) for pr in policy_results)
    route_switch_rates = [
        pr.route_switch_count / max(1, len(pr.choices) - 1) if len(pr.choices) > 1 else 0.0
        for pr in policy_results
    ]

    return {
        "mean_regret": float(regrets.mean()),
        "median_regret": float(np.median(regrets)),
        "tail_regret_p95": float(np.quantile(regrets, 0.95)),
        "total_violation_count": float(sum(pr.violation_count for pr in policy_results)),
        "total_violation_magnitude": float(sum(pr.violation_magnitude for pr in policy_results)),
        "total_deferrals": float(sum(pr.deferral_count for pr in policy_results)),
        "total_avoidable_deferrals": float(
            sum(pr.avoidable_deferral_count for pr in policy_results)
        ),
        "mean_route_switch_rate": float(np.mean(route_switch_rates)),
        "total_depleted_budget_events": float(
            sum(pr.depleted_budget_events for pr in policy_results)
        ),
        "utility_per_resource_unit": (
            float(total_utility / total_consumption) if total_consumption > 0 else float("nan")
        ),
        "n_sequences": float(len(policy_results)),
    }


def paired_regret_deltas(
    a_results: Sequence[PolicyRunResult],
    b_results: Sequence[PolicyRunResult],
    hindsight_results: Dict[str, HindsightResult],
) -> List[float]:
    """Per-sequence (regret_a - regret_b), for paired significance testing.
    Positive means arm A has MORE regret (worse) than arm B on that sequence."""
    b_by_id = {pr.sequence_id: pr for pr in b_results}
    deltas = []
    for pr_a in a_results:
        pr_b = b_by_id[pr_a.sequence_id]
        hindsight = hindsight_results[pr_a.sequence_id]
        regret_a = hindsight.value - pr_a.cumulative_utility
        regret_b = hindsight.value - pr_b.cumulative_utility
        deltas.append(regret_a - regret_b)
    return deltas


def bootstrap_ci(
    deltas: Sequence[float], n_resamples: int = 2000, seed: int = 0
) -> Dict[str, float]:
    """Percentile bootstrap CI on the mean of ``deltas`` (paired regret
    differences). A CI entirely below 0 means A has significantly less
    regret than B; entirely above 0 means A has significantly more."""
    if not deltas:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    arr = np.array(deltas)
    means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)]
    )
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }
