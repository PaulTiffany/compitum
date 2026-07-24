"""Optional inference-trajectory observation for Compitum (observation-only).

This subpackage is dependency-free: importing it (or ``compitum`` itself)
never imports FabricPC or JAX. Concrete external observers live behind
optional adapters; the core defines the protocol, evidence types, a
deterministic no-op observer, and pure-stdlib trajectory sensors.

Trajectory evidence is a companion artifact. It never modifies the frozen
``SwitchCertificate`` schema and never affects route selection in this
tranche.
"""

from .blockwise import blockwise_audit
from .capability import fabricpc_capability
from .noop import NoOpTrajectoryObserver
from .observer import FabricPCTrajectoryObserver
from .protocol import TrajectoryObserver
from .sensors import orientation_audit, second_order_audit
from .types import (
    ObservationStatus,
    TrajectoryEvidence,
    TrajectoryRequest,
    config_sha256,
)

__all__ = [
    "FabricPCTrajectoryObserver",
    "NoOpTrajectoryObserver",
    "ObservationStatus",
    "TrajectoryEvidence",
    "TrajectoryObserver",
    "TrajectoryRequest",
    "blockwise_audit",
    "config_sha256",
    "fabricpc_capability",
    "orientation_audit",
    "second_order_audit",
]
