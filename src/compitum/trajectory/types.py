"""Dependency-free data contracts for trajectory observation.

Python 3.9-compatible, stdlib-only. Nothing here imports FabricPC or JAX.

Semantic boundaries (see docs/adr/0001-fabricpc-trajectory-observer.md):
external node energies recorded here are NOT Compitum's Symbolic Free
Energy, utility, constraint slack, or ``shadow_prices``. Evidence is a
companion artifact and never alters route selection in the observation
tranche.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class ObservationStatus:
    """Closed status vocabulary for a trajectory observation."""

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"
    REFUSED = "refused"
    INVALID = "invalid"
    FAILED = "failed"

    ALL = (OBSERVED, UNAVAILABLE, REFUSED, INVALID, FAILED)


def config_sha256(config: Dict[str, Any]) -> str:
    """Canonical hash of a JSON-serializable configuration mapping."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class TrajectoryRequest:
    """Inputs to a trajectory observer.

    ``config`` must be JSON-serializable; it is hashed canonically so evidence
    can be tied to the exact observation configuration.
    """

    case_id: str
    seeds: Dict[str, int] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    features: Optional[List[float]] = None

    def config_hash(self) -> str:
        return config_sha256({"case_id": self.case_id, "seeds": self.seeds, "config": self.config})


_NONCLAIMS = (
    "directional gain is a finite-difference observation, not a global Lipschitz constant",
    "external node energy is not Compitum Symbolic Free Energy, utility, or constraint slack",
    "no field here is a shadow price or an online dual variable",
    "full-state norm growth is not instability until blockwise transport is audited",
    "observation-only: this evidence does not alter route selection",
)


@dataclass
class TrajectoryEvidence:
    """Result of one trajectory observation (companion artifact contract).

    A non-``observed`` status always carries a ``reason`` and leaves the
    trajectory fields empty: there are no partially populated successes.
    """

    status: str
    observer: str
    observer_version: str
    request_case_id: str
    config_sha256: str
    reason: Optional[str] = None
    dependency_repository: Optional[str] = None
    dependency_commit: Optional[str] = None
    raw_trace_reference: Optional[str] = None
    raw_trace_sha256: Optional[str] = None
    terminal: Dict[str, float] = field(default_factory=dict)
    energy_trajectory: List[float] = field(default_factory=list)
    latent_grad_trajectory: List[float] = field(default_factory=list)
    error_trajectory: List[float] = field(default_factory=list)
    per_step: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    per_node: Dict[str, Dict[str, float]] = field(default_factory=dict)
    convergence: Dict[str, float] = field(default_factory=dict)
    perturbation_diagnostics: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    nonclaims: List[str] = field(default_factory=lambda: list(_NONCLAIMS))
    runtime_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in ObservationStatus.ALL:
            raise ValueError(f"unknown observation status: {self.status!r}")
        if self.status != ObservationStatus.OBSERVED and not self.reason:
            raise ValueError(f"status {self.status!r} requires a reason")
        for name, series in (
            ("energy_trajectory", self.energy_trajectory),
            ("latent_grad_trajectory", self.latent_grad_trajectory),
            ("error_trajectory", self.error_trajectory),
        ):
            if not all(math.isfinite(x) for x in series):
                raise ValueError(f"{name} contains a non-finite value")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = "compitum.trajectory-evidence/v1"
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def evidence_sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
