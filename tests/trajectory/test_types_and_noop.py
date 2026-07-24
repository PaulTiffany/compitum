import json

import pytest

from compitum.trajectory import (
    NoOpTrajectoryObserver,
    ObservationStatus,
    TrajectoryEvidence,
    TrajectoryRequest,
    config_sha256,
)


def test_config_hash_is_canonical_and_order_independent() -> None:
    a = config_sha256({"b": 1, "a": {"y": 2, "x": 3}})
    b = config_sha256({"a": {"x": 3, "y": 2}, "b": 1})
    assert a == b
    assert len(a) == 64


def test_request_hash_covers_case_seeds_and_config() -> None:
    base = TrajectoryRequest(case_id="c1", seeds={"s": 1}, config={"k": 2})
    assert (
        base.config_hash()
        != TrajectoryRequest(case_id="c2", seeds={"s": 1}, config={"k": 2}).config_hash()
    )
    assert (
        base.config_hash()
        != TrajectoryRequest(case_id="c1", seeds={"s": 9}, config={"k": 2}).config_hash()
    )
    assert (
        base.config_hash()
        != TrajectoryRequest(case_id="c1", seeds={"s": 1}, config={"k": 9}).config_hash()
    )
    assert (
        base.config_hash()
        == TrajectoryRequest(case_id="c1", seeds={"s": 1}, config={"k": 2}).config_hash()
    )


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown observation status"):
        TrajectoryEvidence(
            status="wonderful",
            observer="x",
            observer_version="1",
            request_case_id="c",
            config_sha256="0" * 64,
        )


def test_non_observed_status_requires_reason() -> None:
    with pytest.raises(ValueError, match="requires a reason"):
        TrajectoryEvidence(
            status=ObservationStatus.FAILED,
            observer="x",
            observer_version="1",
            request_case_id="c",
            config_sha256="0" * 64,
        )


def test_non_finite_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError, match="energy_trajectory contains a non-finite value"):
        TrajectoryEvidence(
            status=ObservationStatus.OBSERVED,
            observer="x",
            observer_version="1",
            request_case_id="c",
            config_sha256="0" * 64,
            energy_trajectory=[1.0, float("nan")],
        )


def test_evidence_serialization_carries_schema_and_nonclaims() -> None:
    evidence = TrajectoryEvidence(
        status=ObservationStatus.OBSERVED,
        observer="x",
        observer_version="1",
        request_case_id="c",
        config_sha256="0" * 64,
        energy_trajectory=[2.0, 1.0],
    )
    payload = json.loads(evidence.to_json())
    assert payload["schema"] == "compitum.trajectory-evidence/v1"
    assert any("shadow price" in claim for claim in payload["nonclaims"])
    assert any("route selection" in claim for claim in payload["nonclaims"])
    assert len(evidence.evidence_sha256()) == 64


def test_noop_observer_is_deterministic_and_unavailable() -> None:
    observer = NoOpTrajectoryObserver()
    request = TrajectoryRequest(case_id="case-7", seeds={"seed": 7})
    first = observer.observe(request)
    second = observer.observe(request)
    assert first.status == ObservationStatus.UNAVAILABLE
    assert first.reason is not None and "no-op" in first.reason
    assert first.runtime_seconds == 0.0
    assert first.energy_trajectory == []
    assert first.to_json() == second.to_json()
    assert first.evidence_sha256() == second.evidence_sha256()
