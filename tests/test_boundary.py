from typing import Dict

import numpy as np
import pytest

from compitum.boundary import BoundaryAnalyzer


@pytest.mark.parametrize(
    "utilities, u_sigma, expected_is_boundary, reason",
    [
        # High uncertainty, small gap -> boundary
        ({"m1": 0.52, "m2": 0.50}, {"m1": 0.15}, True, "small gap"),
        # High uncertainty, high entropy -> boundary
        ({"m1": 0.52, "m2": 0.48, "m3": 0.45}, {"m1": 0.15}, True, "high entropy"),
        # Low uncertainty -> not boundary
        ({"m1": 0.52, "m2": 0.50}, {"m1": 0.05}, False, "low uncertainty"),
        # Large gap -> not boundary
        ({"m1": 0.8, "m2": 0.5}, {"m1": 0.10}, False, "large gap"),
        # Exact boundary condition: gap < 0.05
        ({"m1": 0.549, "m2": 0.50}, {"m1": 0.13}, True, "gap just inside boundary"),
        # Exact boundary condition: sigma > 0.12
        ({"m1": 0.52, "m2": 0.50}, {"m1": 0.121}, True, "sigma just inside boundary"),
        # Exact boundary condition: entropy > 0.65
        ({"m1": 1.0, "m2": 0.9, "m3": 0.8}, {"m1": 0.13}, True, "entropy just inside boundary"),
    ],
)
def test_boundary_conditions(
    utilities: Dict[str, float],
    u_sigma: Dict[str, float],
    expected_is_boundary: bool,
    reason: str,
) -> None:
    """Test various boundary conditions to kill comparison operator mutants."""
    b = BoundaryAnalyzer()
    info = b.analyze(utilities, u_sigma)
    assert info["is_boundary"] is expected_is_boundary, f"Failed on: {reason}"
    assert info["winner"] == "m1"
    if len(utilities) > 1:
        assert np.isclose(info["utility_gap"], utilities["m1"] - utilities["m2"])


def test_boundary_insufficient_models() -> None:
    b = BoundaryAnalyzer()
    utilities = {"fast": 0.50}
    u_sigma = {"fast": 0.05}
    info = b.analyze(utilities, u_sigma)
    assert info["is_boundary"] is False
    assert info["reason"] == "insufficient_models"
