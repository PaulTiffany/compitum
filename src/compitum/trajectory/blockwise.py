"""Blockwise metric audit for paired trajectories.

Generalized from Sketched's ``fabricpc_orientation_block_sweep.py``
(C:/src/sketched). The load-bearing lesson preserved here: full product-norm
growth can be caused entirely by perturbation transport into another node
block while the originally perturbed block remains contractive. A scalar
full-state gain must therefore never be interpreted as intrinsic instability
without this decomposition. In the Sketched 72-run sweep, 72/72 runs breached
on the product-L2 metric at the first step while 0/72 breached on the
hidden-only metric.

Unlike the Sketched original (which hard-coded a 2+2 hidden/latent layout),
blocks here are declared explicitly as named index ranges.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _norm(values: List[float]) -> float:
    return math.sqrt(sum(x * x for x in values))


def _ratio(after: float, before: float, tolerance: float) -> Optional[float]:
    return None if before <= tolerance else after / before


def blockwise_audit(
    base_states: List[List[float]],
    probe_states: List[List[float]],
    blocks: Dict[str, Tuple[int, int]],
    zero_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Decompose paired-trajectory gains into declared coordinate blocks.

    ``blocks`` maps block name to a half-open index range ``(start, end)``.
    Ranges must be disjoint and cover the full state dimension, so the
    product metric is exactly the norm over all declared blocks.
    """
    if len(base_states) != len(probe_states) or len(base_states) < 2:
        raise ValueError("base and probe trajectories must have equal length >= 2")
    dimension = len(base_states[0])
    if dimension == 0:
        raise ValueError("states must be nonempty")
    if any(len(s) != dimension for s in base_states + probe_states):
        raise ValueError("all states must share one dimension")
    if not blocks:
        raise ValueError("at least one block must be declared")
    covered = [False] * dimension
    for name, (start, end) in blocks.items():
        if not (0 <= start < end <= dimension):
            raise ValueError(f"block {name!r} range ({start}, {end}) is out of bounds")
        for i in range(start, end):
            if covered[i]:
                raise ValueError(f"block {name!r} overlaps another block at index {i}")
            covered[i] = True
    if not all(covered):
        raise ValueError("blocks must cover the full state dimension")
    if not (
        all(math.isfinite(x) for s in base_states for x in s)
        and all(math.isfinite(x) for s in probe_states for x in s)
    ):
        raise ValueError("states contain a non-finite value")

    perturbations = [
        [p - b for p, b in zip(probe, base)] for base, probe in zip(base_states, probe_states)
    ]
    steps: List[Dict[str, Any]] = []
    for step in range(len(perturbations) - 1):
        before = perturbations[step]
        after = perturbations[step + 1]
        block_rows: Dict[str, Dict[str, Any]] = {}
        block_norms_before: List[float] = []
        block_norms_after: List[float] = []
        for name, (start, end) in blocks.items():
            nb = _norm(before[start:end])
            na = _norm(after[start:end])
            block_norms_before.append(nb)
            block_norms_after.append(na)
            block_rows[name] = {
                "before_norm": nb,
                "after_norm": na,
                "gain": _ratio(na, nb, zero_tolerance),
                "emergence": nb <= zero_tolerance and na > zero_tolerance,
            }
        product_before = _norm(before)
        product_after = _norm(after)
        steps.append(
            {
                "step": step,
                "product_l2_gain": _ratio(product_after, product_before, zero_tolerance),
                "max_block_gain": _ratio(
                    max(block_norms_after), max(block_norms_before), zero_tolerance
                ),
                "blocks": block_rows,
            }
        )

    def _breach(value: Optional[float]) -> bool:
        return value is not None and value > 1.0

    first = steps[0]
    return {
        "schema": "compitum.trajectory-blockwise-audit/v1",
        "method": {
            "blocks": {name: list(rng) for name, rng in blocks.items()},
            "metrics": {
                "product_l2": "Euclidean norm on the full concatenated state",
                "max_block": "maximum over declared block Euclidean norms",
                "per_block": "Euclidean norm restricted to each declared block",
            },
            "breach_threshold": 1.0,
            "zero_tolerance": zero_tolerance,
            "interpretation_boundary": (
                "a product-metric breach with no per-block breach indicates "
                "transport across a block interface, not intrinsic expansion"
            ),
        },
        "dimension": dimension,
        "step_count": len(steps),
        "first_step": {
            "product_breach": _breach(first["product_l2_gain"]),
            "max_block_breach": _breach(first["max_block_gain"]),
            "block_breaches": {name: _breach(row["gain"]) for name, row in first["blocks"].items()},
            "block_emergence": {name: row["emergence"] for name, row in first["blocks"].items()},
        },
        "steps": steps,
        "provenance": "generalized from sketched fabricpc_orientation_block_sweep.py",
    }
