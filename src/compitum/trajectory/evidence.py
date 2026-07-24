"""Build governed TrajectoryEvidence from an exported observation payload.

This module is pure: it validates and summarizes an already-exported raw
observation payload (JSON-shaped dicts) and never imports FabricPC or JAX.
The JAX-side exporter lives in ``experiments/fabricpc``; anything it writes
passes through here so every validity rule is enforced in dependency-free,
fully tested code.

Aggregates recorded here are bookkeeping summaries, not physical claims:
``energy_trajectory`` sums per-node energies per step; the gradient/error
trajectories sum per-node mean norms, which is a scalar audit series and not
a single vector norm.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from .types import ObservationStatus, TrajectoryEvidence, TrajectoryRequest

RAW_SCHEMA = "compitum.fabricpc-observation-raw/v1"

_STEP_METRICS = ("energy", "latent_grad_norm", "error_norm")


def _invalid(
    request: TrajectoryRequest,
    observer: str,
    observer_version: str,
    reason: str,
    started: float,
) -> TrajectoryEvidence:
    return TrajectoryEvidence(
        status=ObservationStatus.INVALID,
        observer=observer,
        observer_version=observer_version,
        request_case_id=request.case_id,
        config_sha256=request.config_hash(),
        reason=reason,
        runtime_seconds=time.perf_counter() - started,
    )


def _validate_steps(steps: Any, node_order: List[str]) -> Optional[str]:
    if not isinstance(steps, list) or not steps:
        return "steps must be a nonempty list"
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return f"steps[{index}] must be a mapping of node name to metrics"
        if sorted(step.keys()) != sorted(node_order):
            return (
                f"steps[{index}] node set {sorted(step.keys())} does not match "
                f"declared node_order {sorted(node_order)}"
            )
        for node, metrics in step.items():
            if not isinstance(metrics, dict):
                return f"steps[{index}][{node!r}] must be a metrics mapping"
            for name in _STEP_METRICS:
                value = metrics.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    return f"steps[{index}][{node!r}].{name} is missing or non-finite"
    return None


def build_evidence(
    payload: Dict[str, Any],
    request: TrajectoryRequest,
    observer: str,
    observer_version: str,
    raw_trace_reference: Optional[str] = None,
    raw_trace_sha256: Optional[str] = None,
) -> TrajectoryEvidence:
    """Validate a raw observation payload and produce governed evidence.

    Any malformed input (wrong schema, missing nodes, shape or node-order
    mismatch, non-finite values) yields an ``invalid`` evidence object with a
    structured reason -- never a partially populated success and never an
    uncontrolled exception.
    """
    started = time.perf_counter()

    if not isinstance(payload, dict) or payload.get("schema") != RAW_SCHEMA:
        return _invalid(
            request,
            observer,
            observer_version,
            f"raw payload schema must be {RAW_SCHEMA}",
            started,
        )
    node_order = payload.get("node_order")
    if (
        not isinstance(node_order, list)
        or not node_order
        or not all(isinstance(n, str) for n in node_order)
    ):
        return _invalid(
            request,
            observer,
            observer_version,
            "node_order must be a nonempty list of node names",
            started,
        )
    problem = _validate_steps(payload.get("steps"), node_order)
    if problem is not None:
        return _invalid(request, observer, observer_version, problem, started)

    steps: List[Dict[str, Dict[str, float]]] = [
        {node: {k: float(v) for k, v in metrics.items()} for node, metrics in step.items()}
        for step in payload["steps"]
    ]
    energy = [sum(step[node]["energy"] for node in node_order) for step in steps]
    grad = [sum(step[node]["latent_grad_norm"] for node in node_order) for step in steps]
    error = [sum(step[node]["error_norm"] for node in node_order) for step in steps]

    per_node: Dict[str, Dict[str, float]] = {}
    for node in node_order:
        series = [step[node]["energy"] for step in steps]
        per_node[node] = {
            "terminal_energy": series[-1],
            "mean_energy": sum(series) / len(series),
            "terminal_latent_grad_norm": steps[-1][node]["latent_grad_norm"],
            "terminal_error_norm": steps[-1][node]["error_norm"],
        }

    decreasing = sum(1 for a, b in zip(energy, energy[1:]) if b < a)
    convergence = {
        "steps": float(len(steps)),
        "initial_total_energy": energy[0],
        "terminal_total_energy": energy[-1],
        "energy_reduction_ratio": (energy[-1] / energy[0] if energy[0] != 0.0 else float("inf")),
        "monotone_decreasing_fraction": (
            decreasing / (len(energy) - 1) if len(energy) > 1 else 0.0
        ),
        "terminal_latent_grad_norm_total": grad[-1],
    }
    if not math.isfinite(convergence["energy_reduction_ratio"]):
        convergence["energy_reduction_ratio"] = -1.0

    terminal = {
        "total_energy": energy[-1],
        "total_latent_grad_norm": grad[-1],
        "total_error_norm": error[-1],
    }
    extra_terminal = payload.get("terminal", {})
    if isinstance(extra_terminal, dict):
        for key, value in extra_terminal.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                terminal[str(key)] = float(value)

    runtime = payload.get("runtime_seconds")
    return TrajectoryEvidence(
        status=ObservationStatus.OBSERVED,
        observer=observer,
        observer_version=observer_version,
        request_case_id=request.case_id,
        config_sha256=request.config_hash(),
        dependency_repository=payload.get("dependency_repository"),
        dependency_commit=payload.get("dependency_commit"),
        raw_trace_reference=raw_trace_reference,
        raw_trace_sha256=raw_trace_sha256,
        terminal=terminal,
        energy_trajectory=energy,
        latent_grad_trajectory=grad,
        error_trajectory=error,
        per_step=steps,
        per_node=per_node,
        convergence=convergence,
        perturbation_diagnostics=payload.get("perturbation_diagnostics"),
        runtime_seconds=(
            float(runtime)
            if isinstance(runtime, (int, float)) and math.isfinite(float(runtime))
            else time.perf_counter() - started
        ),
    )
