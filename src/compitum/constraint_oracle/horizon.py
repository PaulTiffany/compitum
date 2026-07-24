"""Sequence-level (horizon) constraint-pressure targets.

Built directly on top of ``static.compute_constraint_pressure`` applied to
each step of an already-computed routing sequence: no new numerical
machinery, just a forward-looking window over the per-step exact slack
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .static import FEASIBILITY_EPSILON, ConstraintPressureResult


@dataclass
class SequenceStepResult:
    """One step of a routing sequence: its case-level oracle result."""

    step: int
    pressure: ConstraintPressureResult


@dataclass
class HorizonOracleResult:
    """Forward-looking targets for one constraint at one step."""

    step: int
    index: int
    binding_within_horizon: bool
    time_to_binding: Optional[int]
    realized_future_slack: float
    horizon: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "index": self.index,
            "binding_within_horizon": self.binding_within_horizon,
            "time_to_binding": self.time_to_binding,
            "realized_future_slack": self.realized_future_slack,
            "horizon": self.horizon,
        }


def compute_horizon_targets(
    steps: List[SequenceStepResult],
    horizon: int,
    epsilon: float = FEASIBILITY_EPSILON,
) -> List[List[HorizonOracleResult]]:
    """For each step ``t`` and constraint ``i``, look ahead up to ``horizon``
    steps (inclusive) within the same sequence.

    ``binding_within_horizon`` is true if constraint ``i`` is violated
    (``current_slack < -epsilon``) at any step in ``[t, t+horizon]``.
    ``time_to_binding`` is the smallest such offset, or ``None`` if it never
    binds within the window. ``realized_future_slack`` is the constraint's
    actual slack at the window's end (``t + horizon``, clamped to the last
    available step) -- the realized value, not a prediction.

    Returns one list of ``HorizonOracleResult`` per step, ordered by
    constraint index, matching ``steps[t].pressure.targets``' order.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not steps:
        return []
    n_constraints = len(steps[0].pressure.targets)
    n_steps = len(steps)
    results: List[List[HorizonOracleResult]] = []
    for t in range(n_steps):
        row: List[HorizonOracleResult] = []
        end = min(t + horizon, n_steps - 1)
        for i in range(n_constraints):
            time_to_binding: Optional[int] = None
            for offset, t_prime in enumerate(range(t, end + 1)):
                slack_t = steps[t_prime].pressure.targets[i].current_slack
                if slack_t < -epsilon:
                    time_to_binding = offset
                    break
            row.append(
                HorizonOracleResult(
                    step=t,
                    index=i,
                    binding_within_horizon=time_to_binding is not None,
                    time_to_binding=time_to_binding,
                    realized_future_slack=steps[end].pressure.targets[i].current_slack,
                    horizon=horizon,
                )
            )
        results.append(row)
    return results
