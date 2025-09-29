from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model

# --- Strategies for generating test data ---


@st.composite
def models_and_utilities_strategy(
    draw: Any, num_models: st.SearchStrategy[int] = st.integers(min_value=1, max_value=5)
) -> Tuple[List[Model], Dict[str, float]]:
    """Generates a list of Models and a corresponding utilities dictionary."""
    n = draw(num_models)
    models = []
    utilities = {}
    for i in range(n):
        name = f"model_{i}"
        # The content of center and capabilities doesn't matter for this test
        model = Model(name=name, center=np.array([]), capabilities=MagicMock(), cost=0.0)
        models.append(model)
        utilities[name] = draw(st.floats(min_value=-100, max_value=100))
    return models, utilities


@st.composite
def constraints_strategy(
    draw: Any, num_constraints: st.SearchStrategy[int] = st.integers(min_value=1, max_value=4)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a constraint system (A, b) and a pgd_banach vector."""
    n_constraints = draw(num_constraints)
    # For this test, we can use a fixed dimension for the Banach vector
    dim_banach = 4
    A = draw(
        st.lists(
            st.lists(st.floats(-1, 1), min_size=dim_banach, max_size=dim_banach),
            min_size=n_constraints,
            max_size=n_constraints,
        )
    )
    b = draw(st.lists(st.floats(-10, 10), min_size=n_constraints, max_size=n_constraints))
    pgd_banach = draw(st.lists(st.floats(-5, 5), min_size=dim_banach, max_size=dim_banach))
    return np.array(A), np.array(b), np.array(pgd_banach)


# --- Invariant Tests ---


@pytest.mark.invariants
@given(
    models_and_utils=models_and_utilities_strategy(),
    constraints=constraints_strategy(),
)
def test_constraint_tightening_monotonicity(
    models_and_utils: tuple[List[Model], Dict[str, float]],
    constraints: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """
    Tests that tightening a constraint never increases the set of viable models.
    This is a metamorphic test.
    POWER: ReplaceComparisonOperator_LtE_Gt, ReplaceComparisonOperator_LtE_Lt
    """
    models, utilities = models_and_utils
    A, b, pgd_banach = constraints

    # --- Original Run ---
    solver1 = ReflectiveConstraintSolver(A, b)
    solver1.select(pgd_banach, models, utilities)
    original_viable_models = {m.name for m in solver1.last_viable_models}

    # --- Metamorphic Run (Tighter Constraint) ---
    assume(b.size > 0)
    # Select a random constraint to tighten
    idx_to_tighten = np.random.randint(0, b.size)
    b_tight = b.copy()
    # Tighten the constraint by a small amount
    b_tight[idx_to_tighten] -= 0.5

    solver2 = ReflectiveConstraintSolver(A, b_tight)
    solver2.select(pgd_banach, models, utilities)
    new_viable_models = {m.name for m in solver2.last_viable_models}

    # --- Assert Invariant ---
    # The new set of viable models must be a subset of the original set.
    assert new_viable_models.issubset(original_viable_models)
