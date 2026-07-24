"""Exact, per-case constraint-pressure oracle.

Structural facts this module relies on (verified by reading the frozen
``ReflectiveConstraintSolver`` in ``src/compitum/constraints.py``, commit
``a8de8cb`` / tag ``v0.2.0``, and confirmed empirically against a numeric
bisection cross-check in ``tests/constraint_oracle/test_static_oracle.py``):

1. ``_is_feasible``'s linear check ``self.A @ xB <= self.b + 1e-10`` uses the
   SAME ``xB`` for every model -- model identity never enters it. Only
   ``model.capabilities.supports(...)`` varies per model. The constraint
   -slack vector ``b - A @ xB`` is therefore identical across the whole
   model pool for a given case.
2. Because ``_is_feasible`` requires ``np.all(...)`` over every row at once,
   a SINGLE violated constraint makes EVERY model simultaneously linearly
   infeasible -- there is no way for a case to have both "a feasible route
   already exists" and "some constraint is currently violated". Whenever any
   constraint is violated, ``ReflectiveConstraintSolver.select`` is
   unconditionally in its ``infeasible_fallback`` branch (``viable == []``,
   returns ``sorted_models[0]`` regardless of utility or capability).

Consequently there are exactly two live cases, not four: either (a) no
constraint is violated, and feasibility is then decided purely by
capability, or (b) some constraint is violated, and the case is
categorically infeasible regardless of capability. This makes critical
relaxation an exact closed-form quantity -- a single number per constraint,
per case -- rather than something requiring adaptive search.

Nothing here is named ``shadow_price`` or "dual variable". The existing
``constraints.shadow_prices`` field (a fixed 1e-5 finite-difference probe)
is read here only as a reference comparison arm, never as ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import Model

FEASIBILITY_EPSILON = 1e-10


@dataclass
class ConstraintTarget:
    """Oracle targets for one constraint index, for one routing case."""

    index: int
    current_slack: float
    already_violated: bool
    near_binding: bool
    reason: str
    critical_relaxation: Optional[float] = None
    marginal_utility_improvement: Optional[float] = None
    best_suppressed_competitor: Optional[str] = None
    discontinuous_winner_change: bool = False
    tied_competitors: List[str] = field(default_factory=list)
    recovers_feasibility: bool = False
    blocking_constraint_indices: List[int] = field(default_factory=list)
    marginal_utility_curve: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "current_slack": self.current_slack,
            "already_violated": self.already_violated,
            "near_binding": self.near_binding,
            "reason": self.reason,
            "critical_relaxation": self.critical_relaxation,
            "marginal_utility_improvement": self.marginal_utility_improvement,
            "best_suppressed_competitor": self.best_suppressed_competitor,
            "discontinuous_winner_change": self.discontinuous_winner_change,
            "tied_competitors": list(self.tied_competitors),
            "recovers_feasibility": self.recovers_feasibility,
            "blocking_constraint_indices": list(self.blocking_constraint_indices),
            "marginal_utility_curve": [list(p) for p in self.marginal_utility_curve],
        }


@dataclass
class ConstraintPressureResult:
    """Oracle output for one routing case: one target per constraint index."""

    schema: str = field(default="compitum.constraint-pressure-oracle/v1", init=False)
    feasible: bool = True
    selected_model: Optional[str] = None
    selected_utility: Optional[float] = None
    fallback_only: bool = False
    capability_blocked_high_utility_models: List[str] = field(default_factory=list)
    jointly_blocking_constraint_indices: List[int] = field(default_factory=list)
    targets: List[ConstraintTarget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "feasible": self.feasible,
            "selected_model": self.selected_model,
            "selected_utility": self.selected_utility,
            "fallback_only": self.fallback_only,
            "capability_blocked_high_utility_models": list(
                self.capability_blocked_high_utility_models
            ),
            "jointly_blocking_constraint_indices": list(self.jointly_blocking_constraint_indices),
            "targets": [t.to_dict() for t in self.targets],
        }


def _needed_delta(slack: float, epsilon: float) -> float:
    """Smallest Δ>=0 with ``slack + Δ >= -epsilon`` (i.e. the row becomes
    satisfied): Δ >= -slack - epsilon."""
    return max(0.0, -slack - epsilon)


def compute_constraint_pressure(
    xB: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    models: List[Model],
    utilities: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
    near_binding_threshold: float = 1e-3,
    epsilon: float = FEASIBILITY_EPSILON,
) -> ConstraintPressureResult:
    """Compute exact constraint-pressure targets for one routing case.

    Mirrors ``ReflectiveConstraintSolver.select``'s own model ranking and
    feasibility semantics exactly (same sort order, same epsilon, same
    capabilities-then-linear check), but replaces its fixed 1e-5 probe with
    the exact critical relaxation for each constraint.
    """
    row_values = A @ xB
    slack = b - row_values  # identical across all models: xB is shared
    violated = slack < -epsilon
    violated_indices = [i for i in range(len(b)) if violated[i]]
    linear_feasible = not violated_indices

    sorted_models = sorted(models, key=lambda m: utilities.get(m.name, -math.inf), reverse=True)
    capable = {
        m.name: (
            m.capabilities.supports(xB, context=context)
            if context is not None
            else m.capabilities.supports(xB)
        )
        for m in models
    }

    result = ConstraintPressureResult()
    result.jointly_blocking_constraint_indices = list(violated_indices)

    if linear_feasible:
        capable_models = [m for m in sorted_models if capable[m.name]]
        if capable_models:
            m_star = capable_models[0]
            m_star_utility = utilities.get(m_star.name, -math.inf)
            result.feasible = True
            result.selected_model = m_star.name
            result.selected_utility = m_star_utility
            higher = [m for m in sorted_models if utilities.get(m.name, -math.inf) > m_star_utility]
            result.capability_blocked_high_utility_models = [
                m.name for m in higher if not capable[m.name]
            ]
        else:
            # Linear-feasible, but every model fails capabilities: still the
            # frozen infeasible_fallback branch.
            result.feasible = False
            result.fallback_only = True
            result.selected_model = sorted_models[0].name if sorted_models else None
    else:
        # Any violation forces the shared _is_feasible check to fail for
        # every model at once, regardless of capability: unconditionally
        # infeasible_fallback.
        result.feasible = False
        result.fallback_only = True
        result.selected_model = sorted_models[0].name if sorted_models else None

    for i in range(len(b)):
        current_slack = float(slack[i])
        already_violated = bool(violated[i])
        near_binding = abs(current_slack) < near_binding_threshold

        if not already_violated:
            # Not currently binding. This holds whether the case is fully
            # feasible (nothing is violated anywhere) or infeasible due to a
            # DIFFERENT constraint: either way, relaxing a row that isn't
            # itself violated cannot change the outcome.
            target = ConstraintTarget(
                index=i,
                current_slack=current_slack,
                already_violated=already_violated,
                near_binding=near_binding,
                reason="not_currently_binding",
            )
        elif len(violated_indices) > 1:
            others = [j for j in violated_indices if j != i]
            target = ConstraintTarget(
                index=i,
                current_slack=current_slack,
                already_violated=already_violated,
                near_binding=near_binding,
                reason="blocked_by_other_constraint",
                blocking_constraint_indices=others,
            )
        else:
            # The sole currently-violated constraint. Relaxing it by
            # delta_needed makes the whole case linear-feasible again,
            # recovering the frozen infeasible_fallback into a genuine
            # capability-gated pick.
            delta_needed = _needed_delta(current_slack, epsilon)
            eligible = [m for m in sorted_models if capable[m.name]]
            if not eligible:
                target = ConstraintTarget(
                    index=i,
                    current_slack=current_slack,
                    already_violated=already_violated,
                    near_binding=near_binding,
                    reason="capability_blocked_only",
                )
            else:
                best_utility = max(utilities.get(m.name, -math.inf) for m in eligible)
                tied = [
                    m.name for m in eligible if utilities.get(m.name, -math.inf) == best_utility
                ]
                best_name = max(
                    eligible, key=lambda m: (utilities.get(m.name, -math.inf), m.name)
                ).name
                target = ConstraintTarget(
                    index=i,
                    current_slack=current_slack,
                    already_violated=already_violated,
                    near_binding=near_binding,
                    reason="recovers_feasibility",
                    critical_relaxation=delta_needed,
                    best_suppressed_competitor=best_name,
                    recovers_feasibility=True,
                    discontinuous_winner_change=len(tied) > 1,
                    tied_competitors=tied,
                    marginal_utility_curve=[(delta_needed, best_utility)],
                )
        result.targets.append(target)

    return result
