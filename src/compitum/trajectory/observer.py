"""Optional FabricPC-backed observer (dependency-injected, governed).

The class below implements :class:`~compitum.trajectory.protocol.TrajectoryObserver`
without importing FabricPC or JAX: the JAX-side exporter
(``experiments/fabricpc/fabricpc_probe.py``, run under the isolated FabricPC
venv) is injected as a ``runner`` callable producing raw
``compitum.fabricpc-observation-raw/v1`` payloads. Every anticipated failure
mode -- optional dependency missing, receipt/checkout drift, malformed or
non-finite payloads, runner crashes -- becomes a structured non-``observed``
evidence object rather than an exception or a partial success.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .capability import fabricpc_capability, verify_receipt
from .evidence import build_evidence
from .types import ObservationStatus, TrajectoryEvidence, TrajectoryRequest

Runner = Callable[[TrajectoryRequest], Dict[str, Any]]


class FabricPCTrajectoryObserver:
    """Observation-only FabricPC trajectory observer.

    ``runner`` maps a request to a raw observation payload; it is only called
    once capability detection and (when configured) receipt verification have
    passed. Route selection never consumes this evidence in the observation
    tranche.
    """

    name = "fabricpc"

    def __init__(
        self,
        runner: Runner,
        receipt_path: Optional[Path] = None,
        checkout: Optional[Path] = None,
    ) -> None:
        self._runner = runner
        self._receipt_path = receipt_path
        self._checkout = checkout
        capability = fabricpc_capability()
        self.version = capability.fabricpc_version or "unavailable"

    def _governed(
        self, request: TrajectoryRequest, status: str, reason: str, started: float
    ) -> TrajectoryEvidence:
        return TrajectoryEvidence(
            status=status,
            observer=self.name,
            observer_version=self.version,
            request_case_id=request.case_id,
            config_sha256=request.config_hash(),
            reason=reason,
            runtime_seconds=time.perf_counter() - started,
        )

    def observe(self, request: TrajectoryRequest) -> TrajectoryEvidence:
        started = time.perf_counter()
        capability = fabricpc_capability()
        if not capability.available:
            return self._governed(
                request,
                ObservationStatus.UNAVAILABLE,
                capability.reason or "optional dependency unavailable",
                started,
            )
        if self._receipt_path is not None and self._checkout is not None:
            drift = verify_receipt(self._receipt_path, self._checkout)
            if drift is not None:
                return self._governed(request, ObservationStatus.REFUSED, drift, started)
        try:
            payload = self._runner(request)
        except Exception as exc:  # runner is external code: governed, not crashed
            return self._governed(
                request,
                ObservationStatus.FAILED,
                f"observation runner raised {type(exc).__name__}: {exc}",
                started,
            )
        return build_evidence(payload, request, self.name, self.version)
