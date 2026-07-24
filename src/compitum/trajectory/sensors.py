"""Pure-stdlib trajectory sensors: orientation and second-order audits.

Ported and generalized from Sketched (C:/src/sketched, verification/tools/
``fabricpc_orientation_sensor.py`` and ``second_order_sensor.py``), with the
original tests retained alongside. Changes from the originals: Compitum
schema identifiers, Python 3.9 compatibility (no ``zip(strict=True)``), and
payload-agnostic provenance fields (``dependency_*`` instead of
``fabricpc_*``). Sketched's Principia/Lean authority model is deliberately
NOT imported: no result here claims any correspondence to a proved guard.

Orientation: for paired base/probe trajectories with per-step difference
``d_t``, the directional gain ``||d_{t+1}|| / ||d_t||`` is a finite-difference
observation, not a global Lipschitz constant. A negative orientation cosine
records a represented-direction reversal, not a causal explanation.

Second order: the mixed finite difference ``combined - first - second + base``
is an observable non-additivity witness only. It is not a Hessian, phase
transition, imagination event, or (alone) an operator-order witness; ordered
A-then-B / B-then-A trajectories provide a separate commutator observable.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

ORIENTATION_INPUT_SCHEMA = "compitum.trajectory-orientation-input/v1"
ORIENTATION_OUTPUT_SCHEMA = "compitum.trajectory-orientation-certificate/v1"
SECOND_ORDER_INPUT_SCHEMA = "compitum.trajectory-second-order-input/v1"
SECOND_ORDER_OUTPUT_SCHEMA = "compitum.trajectory-second-order-certificate/v1"

_SQUARE_BRANCHES = ("base_states", "first_states", "second_states", "combined_states")
_ORDER_BRANCHES = ("order_ab_states", "order_ba_states")


def _vector(value: Any, label: str) -> List[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty numeric array")
    try:
        result = [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _sub(left: List[float], right: List[float]) -> List[float]:
    if len(left) != len(right):
        raise ValueError("paired states must have equal dimensions")
    return [x - y for x, y in zip(left, right)]


def _norm(vector: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def _dot(left: List[float], right: List[float]) -> float:
    if len(left) != len(right):
        raise ValueError("transported perturbations must have equal dimensions")
    return sum(x * y for x, y in zip(left, right))


def _canonical_sha256(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def orientation_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Audit paired base/probe trajectories; emit a hash-pinned certificate."""
    if payload.get("schema") != ORIENTATION_INPUT_SCHEMA:
        raise ValueError(f"input schema must be {ORIENTATION_INPUT_SCHEMA}")

    base = [_vector(x, f"base_states[{i}]") for i, x in enumerate(payload["base_states"])]
    probe = [_vector(x, f"probe_states[{i}]") for i, x in enumerate(payload["probe_states"])]
    if len(base) != len(probe) or len(base) < 2:
        raise ValueError("base_states and probe_states must have equal length >= 2")

    dimension = len(base[0])
    if any(len(x) != dimension for x in base + probe):
        raise ValueError("all states must have one common dimension")

    thresholds = payload.get("thresholds", {})
    gain_limit = float(thresholds.get("directional_gain", 1.0))
    orientation_floor = float(thresholds.get("orientation_cosine", 0.0))
    zero_tolerance = float(thresholds.get("zero_tolerance", 1e-12))
    if gain_limit < 0 or not -1 <= orientation_floor <= 1 or zero_tolerance < 0:
        raise ValueError("invalid sensor thresholds")

    perturbations = [_sub(p, b) for p, b in zip(probe, base)]
    transitions: List[Dict[str, Any]] = []
    for step in range(len(perturbations) - 1):
        before = perturbations[step]
        after = perturbations[step + 1]
        before_norm = _norm(before)
        after_norm = _norm(after)
        degenerate = before_norm <= zero_tolerance or after_norm <= zero_tolerance
        gain: Optional[float] = None if before_norm <= zero_tolerance else after_norm / before_norm
        cosine: Optional[float] = (
            None if degenerate else _dot(before, after) / (before_norm * after_norm)
        )
        gain_breach = gain is not None and gain > gain_limit
        orientation_reversal = cosine is not None and cosine < orientation_floor
        candidate = gain_breach or orientation_reversal
        transitions.append(
            {
                "step": step,
                "input_perturbation_norm": before_norm,
                "output_perturbation_norm": after_norm,
                "directional_gain": gain,
                "orientation_cosine": cosine,
                "degenerate_orientation": degenerate,
                "gain_breach": gain_breach,
                "orientation_reversal": orientation_reversal,
                "candidate_transition": candidate,
                "interpretation": (
                    "candidate: audit latent traversal, operator order, and projection"
                    if candidate
                    else "no configured breach observed"
                ),
            }
        )

    return {
        "schema": ORIENTATION_OUTPUT_SCHEMA,
        "source": {
            "input_sha256": _canonical_sha256(payload),
            "dependency_repository": payload.get("dependency_repository"),
            "dependency_commit": payload.get("dependency_commit"),
            "run_id": payload.get("run_id"),
        },
        "method": {
            "quantity": "paired finite-difference directional gain",
            "global_lipschitz_claim": False,
            "imagination_claim": False,
            "thresholds": {
                "directional_gain": gain_limit,
                "orientation_cosine": orientation_floor,
                "zero_tolerance": zero_tolerance,
            },
        },
        "dimension": dimension,
        "transition_count": len(transitions),
        "candidate_count": sum(t["candidate_transition"] for t in transitions),
        "transitions": transitions,
        "provenance": "ported from sketched fabricpc_orientation_sensor.py",
    }


def _trajectory(payload: Dict[str, Any], key: str) -> List[List[float]]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a nonempty trajectory")
    result = [_vector(state, f"{key}[{i}]") for i, state in enumerate(value)]
    dimension = len(result[0])
    if any(len(state) != dimension for state in result):
        raise ValueError(f"{key} has inconsistent state dimensions")
    return result


def _square_residue(
    base: List[float],
    first: List[float],
    second: List[float],
    combined: List[float],
) -> List[float]:
    if not (len(base) == len(first) == len(second) == len(combined)):
        raise ValueError("state vectors must have equal dimensions")
    return [
        both - one - two + origin for origin, one, two, both in zip(base, first, second, combined)
    ]


def _second_order_thresholds(payload: Dict[str, Any]) -> Tuple[float, float]:
    raw = payload.get("thresholds", {})
    if not isinstance(raw, dict):
        raise ValueError("thresholds must be an object")
    residue = float(raw.get("residue_norm", 1e-9))
    commutator = float(raw.get("commutator_norm", residue))
    if not math.isfinite(residue) or not math.isfinite(commutator) or residue < 0 or commutator < 0:
        raise ValueError("residue and commutator thresholds must be finite and nonnegative")
    return residue, commutator


def second_order_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Audit a four-branch perturbation square; emit a hash-pinned certificate."""
    if not isinstance(payload, dict) or payload.get("schema") != SECOND_ORDER_INPUT_SCHEMA:
        raise ValueError(f"input schema must be {SECOND_ORDER_INPUT_SCHEMA}")

    branches = {name: _trajectory(payload, name) for name in _SQUARE_BRANCHES}
    lengths = {len(trajectory) for trajectory in branches.values()}
    dimensions = {len(trajectory[0]) for trajectory in branches.values()}
    if len(lengths) != 1 or len(dimensions) != 1:
        raise ValueError("all four square branches must have matching shapes")

    residue_threshold, commutator_threshold = _second_order_thresholds(payload)
    order_present = [payload.get(name) is not None for name in _ORDER_BRANCHES]
    if any(order_present) and not all(order_present):
        raise ValueError("order_ab_states and order_ba_states must be supplied together")
    order: Dict[str, List[List[float]]] = {}
    if all(order_present):
        order = {name: _trajectory(payload, name) for name in _ORDER_BRANCHES}
        order_lengths = {len(trajectory) for trajectory in order.values()}
        order_dimensions = {len(trajectory[0]) for trajectory in order.values()}
        if (
            len(order_lengths) != 1
            or len(order_dimensions) != 1
            or next(iter(order_lengths)) != next(iter(lengths))
            or next(iter(order_dimensions)) != next(iter(dimensions))
        ):
            raise ValueError("order branches must match the square branch shape")

    rows: List[Dict[str, Any]] = []
    square = [branches[name] for name in _SQUARE_BRANCHES]
    for step, (base, first, second, combined) in enumerate(zip(*square)):
        residue = _square_residue(base, first, second, combined)
        residue_norm = _norm(residue)
        first_delta_norm = _norm(_sub(first, base))
        second_delta_norm = _norm(_sub(second, base))
        scale = max(first_delta_norm, second_delta_norm)
        row: Dict[str, Any] = {
            "step": step,
            "residue": residue,
            "residue_norm": residue_norm,
            "interaction_ratio": residue_norm / scale if scale > 0 else None,
            "second_order_observed": residue_norm > residue_threshold,
        }
        if order:
            commutator = _sub(order["order_ab_states"][step], order["order_ba_states"][step])
            commutator_norm = _norm(commutator)
            row.update(
                {
                    "commutator": commutator,
                    "commutator_norm": commutator_norm,
                    "order_dependence_observed": commutator_norm > commutator_threshold,
                }
            )
        rows.append(row)

    order_measured = bool(order)
    result: Dict[str, Any] = {
        "schema": SECOND_ORDER_OUTPUT_SCHEMA,
        "source": {
            "input_sha256": _canonical_sha256(payload),
            "run_id": payload.get("run_id"),
            "dependency_repository": payload.get("dependency_repository"),
            "dependency_commit": payload.get("dependency_commit"),
        },
        "method": {
            "observable": "mixed finite-difference interaction residue",
            "formula": "F(x+d1+d2)-F(x+d1)-F(x+d2)+F(x)",
            "commutator_formula": "F_A_then_B(x)-F_B_then_A(x)",
            "order_branches_measured": order_measured,
            "residue_threshold": residue_threshold,
            "commutator_threshold": commutator_threshold,
            "global_lipschitz_claim": False,
            "hessian_claim": False,
            "imagination_claim": False,
            "latent_cause_identified": False,
        },
        "dimension": next(iter(dimensions)),
        "step_count": len(rows),
        "steps_with_second_order_residue": sum(row["second_order_observed"] for row in rows),
        "max_residue_norm": max(row["residue_norm"] for row in rows),
        "steps": rows,
        "provenance": "ported from sketched second_order_sensor.py",
    }
    if order_measured:
        result["steps_with_order_dependence"] = sum(
            row["order_dependence_observed"] for row in rows
        )
        result["max_commutator_norm"] = max(row["commutator_norm"] for row in rows)
    else:
        result["steps_with_order_dependence"] = None
        result["max_commutator_norm"] = None
    return result
