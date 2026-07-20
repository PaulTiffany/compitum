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


def test_boundary_gap_threshold_default_is_not_masked_by_sigma_or_smallness() -> None:
    """Every parametrized case above either has `gap < 0.05` (so a mutated,
    much-larger default threshold like 1.05 would still classify the same
    way) or has `sigma <= sigma_threshold` (masking the gap term entirely
    via the outer `and`). Use a gap that's clearly >= 0.05 but well under
    1.05, combined with sigma > sigma_threshold, so the real default
    (0.05) and a mutated default (1.05) disagree on `is_boundary`."""
    b = BoundaryAnalyzer()
    # gap = 0.7: >= 0.05 (correct threshold) but < 1.05 (the mutant's), and
    # large enough that entropy (~0.635) also stays under entropy_threshold
    # (0.65) so that condition doesn't independently force is_boundary True.
    utilities = {"m1": 0.7, "m2": 0.0}
    info = b.analyze(utilities, u_sigma={"m1": 0.20})  # > sigma_threshold (0.12)
    assert info["is_boundary"] is False


def test_boundary_entropy_exact_value() -> None:
    """No existing test asserts `info["entropy"]`'s actual numeric value --
    only whether it crosses `entropy_threshold`, which several of the
    formula's own possible arithmetic mutations (`*`->`/`, epsilon +/-1e-12)
    don't happen to flip for the parametrized cases above. Recompute the
    exact expected value via the same formula and require bit-exact
    equality (the differences from an epsilon-sized mutation are ~1e-9,
    well inside np.isclose's default tolerance -- too loose to catch it)."""
    b = BoundaryAnalyzer()
    info = b.analyze({"m1": 1.0, "m2": 0.0}, u_sigma={"m1": 0.0})
    arr = np.array([1.0, 0.0])
    probs = np.exp(arr - 1.0)
    probs /= probs.sum()
    expected_entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    assert info["entropy"] == expected_entropy


def test_boundary_gap_threshold_exact_equality_is_not_boundary() -> None:
    """The parametrized cases above test `gap` clearly inside/outside the
    threshold, never exactly equal to it -- `gap < threshold` (correct,
    not-boundary at equality) vs `gap <= threshold` (mutant, boundary at
    equality) were never distinguished. entropy_threshold/sigma_threshold
    are set so only the gap clause is at stake."""
    m1, m2 = 0.6, 0.5
    gap = m1 - m2
    b = BoundaryAnalyzer(gap_threshold=gap, entropy_threshold=0.99, sigma_threshold=0.05)
    info = b.analyze({"m1": m1, "m2": m2}, u_sigma={"m1": 0.5})
    assert info["is_boundary"] is False


def test_boundary_entropy_threshold_exact_equality_is_not_boundary() -> None:
    """Same gap (no pun intended) as above, for `entropy > threshold` vs
    `>=`. gap_threshold is set low enough that the gap clause can't
    independently trigger, isolating the entropy comparison."""
    m1, m2 = 0.6, 0.5
    arr = np.array([m1, m2])
    probs = np.exp(arr - m1)
    probs /= probs.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    b = BoundaryAnalyzer(gap_threshold=0.001, entropy_threshold=entropy, sigma_threshold=0.05)
    info = b.analyze({"m1": m1, "m2": m2}, u_sigma={"m1": 0.5})
    assert info["is_boundary"] is False


def test_boundary_sigma_threshold_exact_equality_is_not_boundary() -> None:
    """Same gap as above, for `sigma > threshold` vs `>=`. gap_threshold is
    set very high so the outer `or` is trivially satisfied regardless,
    isolating the sigma comparison (the outer `and`'s other operand)."""
    b = BoundaryAnalyzer(gap_threshold=10.0, entropy_threshold=0.65, sigma_threshold=0.12)
    info = b.analyze({"m1": 0.6, "m2": 0.5}, u_sigma={"m1": 0.12})
    assert info["is_boundary"] is False


def test_boundary_sigma_defaults_to_zero_when_winner_missing_from_u_sigma() -> None:
    """No other test omits the winner's key from u_sigma, so the `.get(m1, 0.0)`
    default is never exercised -- a mutant changing that default would survive.
    Gap is small enough to satisfy the other half of the boundary condition, so
    whether is_boundary ends up True or False hinges purely on the default."""
    b = BoundaryAnalyzer()
    utilities = {"m1": 0.52, "m2": 0.50}  # gap = 0.02 < gap_threshold (0.05)
    info = b.analyze(utilities, u_sigma={})  # "m1" absent -> sigma defaults to 0.0
    assert info["uncertainty"] == 0.0
    assert info["is_boundary"] is False  # 0.0 is not > sigma_threshold (0.12)
