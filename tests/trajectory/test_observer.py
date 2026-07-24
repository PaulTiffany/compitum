import json
import subprocess
from pathlib import Path

import pytest

import compitum.trajectory.capability as capability_module
import compitum.trajectory.observer as observer_module
from compitum.trajectory import FabricPCTrajectoryObserver, ObservationStatus, TrajectoryRequest
from compitum.trajectory.evidence import RAW_SCHEMA


def _request() -> TrajectoryRequest:
    return TrajectoryRequest(case_id="obs-1", seeds={"parameter_seed": 17})


def _stub_available(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"fabricpc": "0.3.2", "jax": "0.10.2"}
    monkeypatch.setattr(capability_module, "_pkg_version", lambda name: versions[name])


def _raw_payload() -> dict:
    step = {
        "hidden": {
            "energy": 1.0,
            "latent_grad_norm": 0.5,
            "error_norm": 0.25,
        }
    }
    return {
        "schema": RAW_SCHEMA,
        "node_order": ["hidden"],
        "steps": [step, step],
        "dependency_repository": "https://example.invalid/fabricpc",
        "dependency_commit": "a" * 40,
        "runtime_seconds": 0.1,
    }


def test_unavailable_dependency_is_governed() -> None:
    # The plain venv has no fabricpc/jax installed: the real capability path.
    observer = FabricPCTrajectoryObserver(runner=lambda request: _raw_payload())
    evidence = observer.observe(_request())
    assert evidence.status == ObservationStatus.UNAVAILABLE
    assert "not installed" in (evidence.reason or "")
    assert observer.version == "unavailable"


def test_available_runner_success_builds_observed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_available(monkeypatch)
    observer = FabricPCTrajectoryObserver(runner=lambda request: _raw_payload())
    evidence = observer.observe(_request())
    assert evidence.status == ObservationStatus.OBSERVED
    assert evidence.observer == "fabricpc"
    assert evidence.observer_version == "0.3.2"
    assert evidence.energy_trajectory == [1.0, 1.0]


def test_receipt_drift_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_available(monkeypatch)
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
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"source": {"commit": "f" * 40}}), encoding="utf-8")

    calls = []

    def runner(request: TrajectoryRequest) -> dict:
        calls.append(request.case_id)
        return _raw_payload()

    observer = FabricPCTrajectoryObserver(runner=runner, receipt_path=receipt, checkout=repo)
    evidence = observer.observe(_request())
    assert evidence.status == ObservationStatus.REFUSED
    assert "checkout drift" in (evidence.reason or "")
    assert calls == []  # the runner must never execute against a drifted checkout


def test_runner_exception_is_governed_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_available(monkeypatch)

    def runner(request: TrajectoryRequest) -> dict:
        raise RuntimeError("jax exploded")

    observer = FabricPCTrajectoryObserver(runner=runner)
    evidence = observer.observe(_request())
    assert evidence.status == ObservationStatus.FAILED
    assert "RuntimeError: jax exploded" in (evidence.reason or "")


def test_malformed_runner_payload_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_available(monkeypatch)
    observer = FabricPCTrajectoryObserver(runner=lambda request: {"schema": "nope"})
    evidence = observer.observe(_request())
    assert evidence.status == ObservationStatus.INVALID


def test_observer_module_does_not_import_jax() -> None:
    assert not hasattr(observer_module, "jax")


def test_matching_receipt_proceeds_to_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_available(monkeypatch)
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
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"source": {"commit": head}}), encoding="utf-8")

    observer = FabricPCTrajectoryObserver(
        runner=lambda request: _raw_payload(), receipt_path=receipt, checkout=repo
    )
    evidence = observer.observe(_request())
    assert evidence.status == ObservationStatus.OBSERVED
