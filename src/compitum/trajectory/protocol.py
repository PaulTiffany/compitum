"""Observer protocol. Dependency-free; concrete adapters plug in behind it."""

from __future__ import annotations

from typing import Protocol

from .types import TrajectoryEvidence, TrajectoryRequest


class TrajectoryObserver(Protocol):
    """A source of inference-trajectory evidence.

    Implementations must be deterministic given the request's seeds and
    config, must never raise for anticipated unavailability (return a
    governed non-``observed`` evidence instead), and must not import optional
    dependencies at module import time.
    """

    name: str
    version: str

    def observe(self, request: TrajectoryRequest) -> TrajectoryEvidence:
        """Produce evidence for one case; never route-affecting."""
        ...
