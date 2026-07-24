"""Independent, Compitum-owned oracle for constraint pressure (tranche 2).

This package answers a narrower question than tranche 1's generic trajectory
experiment: given a routing case, how close is each linear constraint to
changing the winning route, and by how much would relaxing it improve the
achieved utility? It exists so that any later learned (FabricPC or
otherwise) predictor of "constraint pressure" has an exact, independently
derived ground truth to be judged against -- never the existing finite
-difference ``shadow_prices`` diagnostic, which is one baseline probe (a
fixed 1e-5 bump), not a target authority.

Naming discipline (see docs/adr/0002-constraint-pressure-oracle.md):
nothing in this package is called a shadow price or a dual variable. Fields
use ``critical_relaxation``, ``constraint_pressure``, and similar explicitly
experimental names. ``src/compitum/constraints.py`` and its
``shadow_prices`` field are read-only references here, never modified.

This package is dependency-free beyond what the frozen core already uses
(numpy); it never imports FabricPC or JAX.
"""

from .horizon import (
    HorizonOracleResult,
    SequenceStepResult,
    compute_horizon_targets,
)
from .static import (
    ConstraintPressureResult,
    ConstraintTarget,
    compute_constraint_pressure,
)

__all__ = [
    "ConstraintPressureResult",
    "ConstraintTarget",
    "HorizonOracleResult",
    "SequenceStepResult",
    "compute_constraint_pressure",
    "compute_horizon_targets",
]
