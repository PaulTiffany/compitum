"""Deterministic no-op observer establishing exact baseline behavior."""

from __future__ import annotations

from .types import ObservationStatus, TrajectoryEvidence, TrajectoryRequest


class NoOpTrajectoryObserver:
    """Observes nothing, deterministically.

    Exists so the experiment harness can run identically shaped code in the
    baseline arm: the returned evidence is byte-stable for a given request
    (status ``unavailable``, fixed reason, no trajectories), and routing
    behavior with this observer must equal frozen v0.2.0 exactly.
    """

    name = "noop"
    version = "1"

    def observe(self, request: TrajectoryRequest) -> TrajectoryEvidence:
        return TrajectoryEvidence(
            status=ObservationStatus.UNAVAILABLE,
            observer=self.name,
            observer_version=self.version,
            request_case_id=request.case_id,
            config_sha256=request.config_hash(),
            reason="no-op observer: trajectory observation intentionally disabled",
            runtime_seconds=0.0,
        )
