import os
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from hypothesis import strategies as st


@dataclass(frozen=True)
class Tolerances:
    """Centralizes numeric tolerances for invariant testing."""
    rel: float = 1e-7
    abs: float = 1e-9

TOL = Tolerances()

def get_active_profile() -> str:
    """Returns the name of the currently active Hypothesis profile."""
    return os.getenv("HYPOTHESIS_PROFILE", "ci")

def is_stress_profile() -> bool:
    """Checks if the 'stress' profile is active."""
    return get_active_profile() == "stress"

def boundary_floats(min_value: float = -1e6, max_value: float = 1e6) -> st.SearchStrategy[float]:
    """Generates floats, biased towards boundary conditions."""
    return st.floats(
        min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False
    )

def boundary_grid(
    base_strategy: st.SearchStrategy[float],
) -> st.SearchStrategy[float]:
    """Forces inclusion of boundary values in a float strategy."""
    return (
        base_strategy
        | st.floats(min_value=0, max_value=0)
        | st.floats(min_value=-1e-9, max_value=1e-9)
    )

@st.composite
def create_metamorphic_pair(
    draw: Any, vec_strategy: st.SearchStrategy[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a pair of vectors for metamorphic testing: the original, and a slightly
    perturbed version (e.g., scaled or with added epsilon).
    """
    original = draw(vec_strategy)
    perturbation = draw(st.floats(min_value=-1e-6, max_value=1e-6))
    perturbed = original * (1 + perturbation)
    return original, perturbed
