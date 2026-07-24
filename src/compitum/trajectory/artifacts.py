"""Companion experiment-artifact bundles for trajectory observations.

Schema family: ``compitum.fabricpc-observation/v1``. Each observation run
produces a bundle directory containing the raw trajectory, the trajectory
evidence certificate, an audit/provenance record, an experiment manifest, a
validation summary, and a checksums file hashing every member. The frozen
``SwitchCertificate`` is never modified; a route certificate and a trajectory
certificate may reference one another by hash only.

Bundle layout, checksums-last so the checksum file can cover every other
member, is an independent clean-room implementation of common
manifest/checksum patterns using Compitum-specific names and schemas.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import ObservationStatus, TrajectoryEvidence

BUNDLE_SCHEMA = "compitum.fabricpc-observation/v1"

RAW_TRACE = "raw_trace.json"
EVIDENCE = "trajectory_evidence.json"
AUDIT = "audit_record.json"
MANIFEST = "experiment_manifest.json"
VALIDATION = "validation_summary.json"
CHECKSUMS = "checksums.json"


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8", newline="")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass
class ObservationBundle:
    directory: Path
    checksums: Dict[str, str]
    bundle_sha256: str


def write_observation_bundle(
    directory: Path,
    evidence: TrajectoryEvidence,
    raw_payload: Optional[Dict[str, Any]],
    manifest_extra: Optional[Dict[str, Any]] = None,
    route_certificate_sha256: Optional[str] = None,
) -> ObservationBundle:
    """Write one observation bundle; returns member hashes and a bundle hash.

    ``raw_payload`` may be ``None`` for non-``observed`` evidence (there is no
    raw trace to retain for a refusal); the validation summary then records
    the governed non-success explicitly.
    """
    directory.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    checksums: Dict[str, str] = {}

    if raw_payload is not None:
        checksums[RAW_TRACE] = _write_json(directory / RAW_TRACE, raw_payload)

    evidence_payload = evidence.to_dict()
    checksums[EVIDENCE] = _write_json(directory / EVIDENCE, evidence_payload)

    audit = {
        "schema": "compitum.fabricpc-observation-audit/v1",
        "generated_at": generated_at,
        "observer": evidence.observer,
        "observer_version": evidence.observer_version,
        "dependency_repository": evidence.dependency_repository,
        "dependency_commit": evidence.dependency_commit,
        "request_case_id": evidence.request_case_id,
        "config_sha256": evidence.config_sha256,
        "raw_trace_reference": evidence.raw_trace_reference,
        "raw_trace_sha256": evidence.raw_trace_sha256,
        "route_certificate_sha256": route_certificate_sha256,
        "frozen_route_certificate_unchanged": True,
    }
    checksums[AUDIT] = _write_json(directory / AUDIT, audit)

    manifest: Dict[str, Any] = {
        "schema": "compitum.fabricpc-observation-manifest/v1",
        "bundle_schema": BUNDLE_SCHEMA,
        "generated_at": generated_at,
        "members": sorted([*checksums.keys(), VALIDATION, CHECKSUMS]),
        "observation_status": evidence.status,
    }
    if manifest_extra:
        manifest["experiment"] = manifest_extra
    checksums[MANIFEST] = _write_json(directory / MANIFEST, manifest)

    problems: List[str] = []
    if evidence.status != ObservationStatus.OBSERVED:
        problems.append(f"governed non-success: {evidence.status}: {evidence.reason}")
    if raw_payload is None and evidence.status == ObservationStatus.OBSERVED:
        problems.append("observed evidence without a retained raw trace")
    validation = {
        "schema": "compitum.fabricpc-observation-validation/v1",
        "generated_at": generated_at,
        "ok": not problems,
        "problems": problems,
        "evidence_sha256": evidence.evidence_sha256(),
    }
    checksums[VALIDATION] = _write_json(directory / VALIDATION, validation)

    checksums_payload = {
        "schema": "compitum.fabricpc-observation-checksums/v1",
        "generated_at": generated_at,
        "algorithm": "sha256",
        "members": dict(sorted(checksums.items())),
    }
    canonical = json.dumps(checksums_payload, sort_keys=True, separators=(",", ":"))
    bundle_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    checksums_payload["bundle_sha256"] = bundle_sha256
    _write_json(directory / CHECKSUMS, checksums_payload)

    return ObservationBundle(directory=directory, checksums=checksums, bundle_sha256=bundle_sha256)
