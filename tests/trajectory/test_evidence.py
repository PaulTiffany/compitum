import math

import pytest

from compitum.trajectory.evidence import RAW_SCHEMA, build_evidence
from compitum.trajectory.types import ObservationStatus, TrajectoryRequest


def _request() -> TrajectoryRequest:
    return TrajectoryRequest(case_id="case-1", seeds={"parameter_seed": 17})


def _step(energy_by_node):
    return {
        node: {
            "energy": energy,
            "latent_grad_norm": energy / 2.0,
            "error_norm": energy / 4.0,
            "z_latent_mean": 0.1,
            "z_latent_std": 0.2,
        }
        for node, energy in energy_by_node.items()
    }


def _payload(**overrides):
    payload = {
        "schema": RAW_SCHEMA,
        "run_id": "r1",
        "case_id": "case-1",
        "dependency_repository": "https://example.invalid/fabricpc",
        "dependency_commit": "a" * 40,
        "node_order": ["source", "hidden"],
        "steps": [
            _step({"source": 0.0, "hidden": 4.0}),
            _step({"source": 0.0, "hidden": 2.0}),
            _step({"source": 0.0, "hidden": 1.0}),
        ],
        "terminal": {"extra_metric": 7.5, "bad": float("nan"), "text": "ignored"},
        "runtime_seconds": 0.25,
    }
    payload.update(overrides)
    return payload


def test_valid_payload_builds_observed_evidence_with_summaries() -> None:
    evidence = build_evidence(
        _payload(),
        _request(),
        "fabricpc",
        "0.3.2",
        raw_trace_reference="bundle/raw_trace.json",
        raw_trace_sha256="b" * 64,
    )
    assert evidence.status == ObservationStatus.OBSERVED
    assert evidence.energy_trajectory == [4.0, 2.0, 1.0]
    assert evidence.latent_grad_trajectory == [2.0, 1.0, 0.5]
    assert evidence.error_trajectory == [1.0, 0.5, 0.25]
    assert evidence.terminal["total_energy"] == 1.0
    assert evidence.terminal["extra_metric"] == 7.5
    assert "bad" not in evidence.terminal and "text" not in evidence.terminal
    assert evidence.per_node["hidden"]["terminal_energy"] == 1.0
    assert evidence.per_node["hidden"]["mean_energy"] == pytest.approx(7.0 / 3.0)
    assert evidence.convergence["steps"] == 3.0
    assert evidence.convergence["energy_reduction_ratio"] == pytest.approx(0.25)
    assert evidence.convergence["monotone_decreasing_fraction"] == 1.0
    assert evidence.dependency_commit == "a" * 40
    assert evidence.raw_trace_sha256 == "b" * 64
    assert evidence.runtime_seconds == 0.25


def test_wrong_schema_yields_governed_invalid() -> None:
    evidence = build_evidence(_payload(schema="nope"), _request(), "f", "1")
    assert evidence.status == ObservationStatus.INVALID
    assert evidence.reason is not None and RAW_SCHEMA in evidence.reason
    assert evidence.energy_trajectory == []
    assert evidence.runtime_seconds is not None


def test_non_dict_payload_is_invalid() -> None:
    assert build_evidence(["nope"], _request(), "f", "1").status == ObservationStatus.INVALID


def test_bad_node_order_is_invalid() -> None:
    for node_order in ([], "hidden", [1, 2]):
        evidence = build_evidence(_payload(node_order=node_order), _request(), "f", "1")
        assert evidence.status == ObservationStatus.INVALID
        assert "node_order" in (evidence.reason or "")


def test_step_node_set_mismatch_is_invalid() -> None:
    payload = _payload()
    payload["steps"][1] = _step({"source": 0.0, "wrong_node": 2.0})
    evidence = build_evidence(payload, _request(), "f", "1")
    assert evidence.status == ObservationStatus.INVALID
    assert "does not match declared node_order" in (evidence.reason or "")


def test_missing_or_nonfinite_metric_is_invalid() -> None:
    payload = _payload()
    del payload["steps"][2]["hidden"]["error_norm"]
    evidence = build_evidence(payload, _request(), "f", "1")
    assert evidence.status == ObservationStatus.INVALID
    assert "missing or non-finite" in (evidence.reason or "")

    payload = _payload()
    payload["steps"][0]["hidden"]["energy"] = math.inf
    assert build_evidence(payload, _request(), "f", "1").status == ObservationStatus.INVALID


def test_malformed_steps_are_invalid() -> None:
    for steps in ([], "steps", [["not", "a", "dict"]]):
        evidence = build_evidence(_payload(steps=steps), _request(), "f", "1")
        assert evidence.status == ObservationStatus.INVALID
    payload = _payload()
    payload["steps"][0]["hidden"] = "not-a-mapping"
    assert build_evidence(payload, _request(), "f", "1").status == ObservationStatus.INVALID


def test_zero_initial_energy_branch_is_finite() -> None:
    payload = _payload(
        steps=[_step({"source": 0.0, "hidden": 0.0}), _step({"source": 0.0, "hidden": 0.0})]
    )
    evidence = build_evidence(payload, _request(), "f", "1")
    assert evidence.status == ObservationStatus.OBSERVED
    assert evidence.convergence["energy_reduction_ratio"] == -1.0
    assert evidence.convergence["monotone_decreasing_fraction"] == 0.0


def test_single_step_and_runtime_fallback() -> None:
    payload = _payload(steps=[_step({"source": 0.0, "hidden": 3.0})])
    payload["runtime_seconds"] = "not-a-number"
    payload["terminal"] = "not-a-dict"
    evidence = build_evidence(payload, _request(), "f", "1")
    assert evidence.status == ObservationStatus.OBSERVED
    assert evidence.convergence["monotone_decreasing_fraction"] == 0.0
    assert evidence.runtime_seconds is not None and evidence.runtime_seconds >= 0.0
    assert evidence.terminal["total_energy"] == 3.0
