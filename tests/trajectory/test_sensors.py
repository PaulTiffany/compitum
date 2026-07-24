"""Sensor tests, retained/extended from their Sketched originals.

Provenance: test_fabricpc_orientation_sensor.py and test_second_order_sensor.py
in C:/src/sketched/verification/tools, adapted to the Compitum schemas.
"""

import pytest

from compitum.trajectory.sensors import (
    ORIENTATION_INPUT_SCHEMA,
    SECOND_ORDER_INPUT_SCHEMA,
    orientation_audit,
    second_order_audit,
)


def _orientation_payload(base, probe, thresholds=None):
    payload = {
        "schema": ORIENTATION_INPUT_SCHEMA,
        "run_id": "test",
        "dependency_repository": "https://example.invalid/fabricpc",
        "dependency_commit": "f" * 40,
        "base_states": base,
        "probe_states": probe,
    }
    if thresholds is not None:
        payload["thresholds"] = thresholds
    return payload


def test_contracting_orientation_preserving_pair_is_clear() -> None:
    base = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    probe = [[1.0, 0.0], [0.5, 0.0], [0.25, 0.0]]
    certificate = orientation_audit(_orientation_payload(base, probe))
    assert certificate["candidate_count"] == 0
    assert certificate["transition_count"] == 2
    assert certificate["source"]["dependency_commit"] == "f" * 40
    assert certificate["method"]["global_lipschitz_claim"] is False


def test_gain_breach_is_candidate() -> None:
    base = [[0.0, 0.0], [0.0, 0.0]]
    probe = [[1.0, 0.0], [2.0, 0.0]]
    certificate = orientation_audit(_orientation_payload(base, probe))
    (transition,) = certificate["transitions"]
    assert transition["gain_breach"] is True
    assert transition["orientation_reversal"] is False
    assert transition["candidate_transition"] is True
    assert "candidate" in transition["interpretation"]


def test_orientation_reversal_is_candidate() -> None:
    base = [[0.0, 0.0], [0.0, 0.0]]
    probe = [[1.0, 0.0], [-0.5, 0.0]]
    certificate = orientation_audit(_orientation_payload(base, probe))
    (transition,) = certificate["transitions"]
    assert transition["orientation_reversal"] is True
    assert transition["gain_breach"] is False


def test_zero_perturbation_does_not_invent_orientation() -> None:
    base = [[0.0, 0.0], [0.0, 0.0]]
    probe = [[0.0, 0.0], [1.0, 0.0]]
    certificate = orientation_audit(_orientation_payload(base, probe))
    (transition,) = certificate["transitions"]
    assert transition["degenerate_orientation"] is True
    assert transition["directional_gain"] is None
    assert transition["orientation_cosine"] is None
    assert transition["candidate_transition"] is False


def test_orientation_schema_and_shape_rejections() -> None:
    with pytest.raises(ValueError, match="input schema"):
        orientation_audit({"schema": "nope"})
    with pytest.raises(ValueError, match="equal length >= 2"):
        orientation_audit(_orientation_payload([[1.0]], [[1.0]]))
    with pytest.raises(ValueError, match="one common dimension"):
        orientation_audit(_orientation_payload([[1.0], [1.0]], [[1.0], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="must be a nonempty numeric array"):
        orientation_audit(_orientation_payload([[], []], [[], []]))
    with pytest.raises(ValueError, match="must be numeric"):
        orientation_audit(_orientation_payload([["x"], [1.0]], [[1.0], [1.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        orientation_audit(_orientation_payload([[float("inf")], [1.0]], [[1.0], [1.0]]))
    with pytest.raises(ValueError, match="invalid sensor thresholds"):
        orientation_audit(
            _orientation_payload(
                [[0.0], [0.0]], [[1.0], [0.5]], thresholds={"directional_gain": -1}
            )
        )


def _square_payload(base, first, second, combined, **extra):
    payload = {
        "schema": SECOND_ORDER_INPUT_SCHEMA,
        "run_id": "square-test",
        "base_states": base,
        "first_states": first,
        "second_states": second,
        "combined_states": combined,
    }
    payload.update(extra)
    return payload


def test_additive_null_has_zero_residue() -> None:
    base = [[0.0, 0.0]]
    first = [[1.0, 0.0]]
    second = [[0.0, 1.0]]
    combined = [[1.0, 1.0]]
    certificate = second_order_audit(_square_payload(base, first, second, combined))
    assert certificate["steps_with_second_order_residue"] == 0
    assert certificate["max_residue_norm"] == 0.0
    assert certificate["method"]["hessian_claim"] is False
    assert certificate["steps_with_order_dependence"] is None
    assert certificate["max_commutator_norm"] is None


def test_bilinear_positive_control_has_residue() -> None:
    # F(a, b) includes an a*b interaction: combined != first + second - base.
    base = [[0.0]]
    first = [[1.0]]
    second = [[1.0]]
    combined = [[3.0]]  # residue = 3 - 1 - 1 + 0 = 1
    certificate = second_order_audit(_square_payload(base, first, second, combined))
    assert certificate["steps_with_second_order_residue"] == 1
    assert certificate["max_residue_norm"] == pytest.approx(1.0)
    (row,) = certificate["steps"]
    assert row["interaction_ratio"] == pytest.approx(1.0)


def test_threshold_is_explicit() -> None:
    base, first, second, combined = [[0.0]], [[1.0]], [[1.0]], [[2.0 + 5e-10]]
    below = second_order_audit(_square_payload(base, first, second, combined))
    assert below["steps_with_second_order_residue"] == 0
    above = second_order_audit(
        _square_payload(base, first, second, combined, thresholds={"residue_norm": 1e-10})
    )
    assert above["steps_with_second_order_residue"] == 1


def test_order_commutator_is_separate_observable() -> None:
    base, first, second, combined = [[0.0]], [[1.0]], [[1.0]], [[2.0]]
    certificate = second_order_audit(
        _square_payload(
            base,
            first,
            second,
            combined,
            order_ab_states=[[5.0]],
            order_ba_states=[[4.0]],
        )
    )
    assert certificate["steps_with_second_order_residue"] == 0
    assert certificate["steps_with_order_dependence"] == 1
    assert certificate["max_commutator_norm"] == pytest.approx(1.0)


def test_partial_order_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        second_order_audit(
            _square_payload([[0.0]], [[1.0]], [[1.0]], [[2.0]], order_ab_states=[[1.0]])
        )


def test_shape_mismatches_are_rejected() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        second_order_audit(_square_payload([[0.0]], [[1.0]], [[1.0]], [[2.0], [2.0]]))
    with pytest.raises(ValueError, match="matching shapes"):
        second_order_audit(_square_payload([[0.0]], [[1.0]], [[1.0]], [[2.0, 0.0]]))
    with pytest.raises(ValueError, match="order branches must match"):
        second_order_audit(
            _square_payload(
                [[0.0]],
                [[1.0]],
                [[1.0]],
                [[2.0]],
                order_ab_states=[[1.0, 0.0]],
                order_ba_states=[[1.0, 0.0]],
            )
        )


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="input schema"):
        second_order_audit({"schema": "nope"})
    with pytest.raises(ValueError, match="input schema"):
        second_order_audit(["not", "a", "dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty trajectory"):
        second_order_audit(_square_payload([], [[1.0]], [[1.0]], [[2.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        second_order_audit(_square_payload([[float("nan")]], [[1.0]], [[1.0]], [[2.0]]))
    with pytest.raises(ValueError, match="inconsistent state dimensions"):
        second_order_audit(
            _square_payload([[0.0], [0.0, 1.0]], [[1.0], [1.0]], [[1.0], [1.0]], [[2.0], [2.0]])
        )
    with pytest.raises(ValueError, match="thresholds must be an object"):
        second_order_audit(_square_payload([[0.0]], [[1.0]], [[1.0]], [[2.0]], thresholds=[1]))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        second_order_audit(
            _square_payload([[0.0]], [[1.0]], [[1.0]], [[2.0]], thresholds={"residue_norm": -1})
        )


def test_private_helper_dimension_guards() -> None:
    """The public audits pre-validate shapes, so these defensive guards are
    exercised directly rather than left as untested dead branches."""
    from compitum.trajectory.sensors import _dot, _square_residue, _sub

    with pytest.raises(ValueError, match="equal dimensions"):
        _sub([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="equal dimensions"):
        _dot([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="equal dimensions"):
        _square_residue([1.0], [1.0], [1.0], [1.0, 2.0])
