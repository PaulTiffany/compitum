"""Numeric cross-check for the closed-form oracle in ``static.py``.

Re-runs the actual frozen ``ReflectiveConstraintSolver.select`` at a grid of
relaxation values to confirm the closed-form critical relaxation agrees with
brute-force search. This exists to validate ``static.py``'s derivation
empirically rather than trust the algebra alone (the same discipline used
throughout this project's mutation-provenance work).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..constraints import ReflectiveConstraintSolver
from ..models import Model


def numeric_critical_relaxation(
    constraint_index: int,
    xB: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    models: List[Model],
    utilities: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
    upper_bound: float = 100.0,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> Optional[float]:
    """Bisection search for the smallest Δ that changes the winning model.

    Returns ``None`` if no relaxation up to ``upper_bound`` changes the
    winner (matching the closed-form's ``None`` for "no relevant
    competitor" or "blocked by another constraint").
    """
    baseline_solver = ReflectiveConstraintSolver(A, b)
    baseline_winner, _ = baseline_solver.select(xB, models, utilities, context=context)

    def winner_at(delta: float) -> str:
        b_relaxed = b.copy()
        b_relaxed[constraint_index] += delta
        solver = ReflectiveConstraintSolver(A, b_relaxed)
        winner, _ = solver.select(xB, models, utilities, context=context)
        return str(winner.name)

    if winner_at(upper_bound) == baseline_winner.name:
        return None

    lo, hi = 0.0, upper_bound
    for _ in range(max_iterations):
        if hi - lo < tolerance:
            break
        mid = (lo + hi) / 2.0
        if winner_at(mid) == baseline_winner.name:
            lo = mid
        else:
            hi = mid
    return hi
