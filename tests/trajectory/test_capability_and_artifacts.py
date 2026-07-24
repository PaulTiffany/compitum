import json
import subprocess
from pathlib import Path

import pytest

import compitum.trajectory.capability as capability_module
from compitum.trajectory import fabricpc_capability
from compitum.trajectory.artifacts import (
    CHECKSUMS,
    EVIDENCE,
    RAW_TRACE,
    VALIDATION,
    write_observation_bundle,
)
from compitum.trajectory.capability import verify_receipt
from compitum.trajectory.types import ObservationStatus, TrajectoryEvidence


def test_capability_unavailable_without_fabricpc_installed() -> None:
    # The plain test venv deliberately has neither fabricpc nor jax.
    result = fabricpc_capability()
    assert result.available is False
    assert result.reason is not None and "not installed" in result.reason
    assert result.fabricpc_version is None


def test_capability_available_with_stubbed_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"fabricpc": "0.3.2", "jax": "0.10.2"}
    monkeypatch.setattr(capability_module, "_pkg_version", lambda name: versions[name])
    result = fabricpc_capability()
    assert result.available is True
    assert result.fabricpc_version == "0.3.2"
    assert result.jax_version == "0.10.2"
    assert result.reason is None


def _receipt(tmp_path: Path, commit: str) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"source": {"commit": commit}}), encoding="utf-8")
    return path


def _git_repo(tmp_path: Path) -> "tuple[Path, str]":
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "pin",
        ],
        check=True,
    )
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return repo, head


def test_verify_receipt_match_and_drift(tmp_path: Path) -> None:
    repo, head = _git_repo(tmp_path)
    assert verify_receipt(_receipt(tmp_path, head), repo) is None
    drift = verify_receipt(_receipt(tmp_path, "f" * 40), repo)
    assert drift is not None and "checkout drift" in drift


def test_verify_receipt_governed_failures(tmp_path: Path) -> None:
    repo, head = _git_repo(tmp_path)
    missing = verify_receipt(tmp_path / "absent.json", repo)
    assert missing is not None and "unreadable receipt" in missing

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    assert "unreadable receipt" in (verify_receipt(bad_json, repo) or "")

    no_commit = tmp_path / "empty.json"
    no_commit.write_text(json.dumps({"source": {}}), encoding="utf-8")
    assert "does not pin a source commit" in (verify_receipt(no_commit, repo) or "")

    not_a_repo = verify_receipt(_receipt(tmp_path, head), tmp_path / "nowhere")
    assert not_a_repo is not None and "cannot resolve checkout HEAD" in not_a_repo


def _observed_evidence() -> TrajectoryEvidence:
    return TrajectoryEvidence(
        status=ObservationStatus.OBSERVED,
        observer="fabricpc",
        observer_version="0.3.2",
        request_case_id="case-1",
        config_sha256="c" * 64,
        dependency_repository="https://example.invalid/fabricpc",
        dependency_commit="a" * 40,
        energy_trajectory=[2.0, 1.0],
    )


def test_observed_bundle_contains_all_members_and_hashes(tmp_path: Path) -> None:
    raw = {"schema": "compitum.fabricpc-observation-raw/v1", "steps": []}
    bundle = write_observation_bundle(
        tmp_path / "b1",
        _observed_evidence(),
        raw,
        manifest_extra={"arm": "trajectory-summary"},
        route_certificate_sha256="d" * 64,
    )
    names = {p.name for p in bundle.directory.iterdir()}
    assert names == {
        "raw_trace.json",
        "trajectory_evidence.json",
        "audit_record.json",
        "experiment_manifest.json",
        "validation_summary.json",
        "checksums.json",
    }
    checks = json.loads((bundle.directory / CHECKSUMS).read_text(encoding="utf-8"))
    assert checks["bundle_sha256"] == bundle.bundle_sha256
    assert set(checks["members"]) == {
        RAW_TRACE,
        EVIDENCE,
        "audit_record.json",
        "experiment_manifest.json",
        VALIDATION,
    }
    validation = json.loads((bundle.directory / VALIDATION).read_text(encoding="utf-8"))
    assert validation["ok"] is True and validation["problems"] == []
    audit = json.loads((bundle.directory / "audit_record.json").read_text(encoding="utf-8"))
    assert audit["route_certificate_sha256"] == "d" * 64
    assert audit["frozen_route_certificate_unchanged"] is True
    manifest = json.loads(
        (bundle.directory / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["experiment"] == {"arm": "trajectory-summary"}


def test_refusal_bundle_has_no_raw_trace_and_is_flagged(tmp_path: Path) -> None:
    refused = TrajectoryEvidence(
        status=ObservationStatus.REFUSED,
        observer="fabricpc",
        observer_version="0.3.2",
        request_case_id="case-1",
        config_sha256="c" * 64,
        reason="FabricPC checkout drift: receipt=aa, actual=bb",
    )
    bundle = write_observation_bundle(tmp_path / "b2", refused, raw_payload=None)
    assert not (bundle.directory / RAW_TRACE).exists()
    validation = json.loads((bundle.directory / VALIDATION).read_text(encoding="utf-8"))
    assert validation["ok"] is False
    assert any("governed non-success: refused" in p for p in validation["problems"])


def test_observed_without_raw_trace_is_a_validation_problem(tmp_path: Path) -> None:
    bundle = write_observation_bundle(tmp_path / "b3", _observed_evidence(), raw_payload=None)
    validation = json.loads((bundle.directory / VALIDATION).read_text(encoding="utf-8"))
    assert validation["ok"] is False
    assert any("without a retained raw trace" in p for p in validation["problems"])


def test_bundle_hash_is_deterministic_for_identical_content(tmp_path: Path) -> None:
    raw = {"schema": "compitum.fabricpc-observation-raw/v1", "steps": []}
    one = write_observation_bundle(tmp_path / "one", _observed_evidence(), raw)
    two = write_observation_bundle(tmp_path / "two", _observed_evidence(), raw)
    # member hashes identical; bundle hash covers members but not generated_at
    assert one.checksums[EVIDENCE] == two.checksums[EVIDENCE]
    assert one.checksums[RAW_TRACE] == two.checksums[RAW_TRACE]
