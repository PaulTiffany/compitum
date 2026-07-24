"""Declared, inspectable channel mapping from Compitum routing/constraint
state to a fixed-layout observation vector (tranche 2).

This is the input FabricPC observes -- never the reverse. Every channel is
computed from state Compitum itself already has (the oracle's own slack/
feasibility computation, the routing case's utilities, and the frozen,
unmodified ``LyapunovController``); nothing here consumes a future answer.

Fixed channel layout (17-dimensional; order and meaning are part of this
module's public contract, not an implementation detail):

    [0:4]   normalized constraint slack, one per constraint index
    [4:7]   feasibility mask by model (1.0/0.0), in ``model_names`` order
    [7:10]  utility by model, in ``model_names`` order
    [10]    winner/runner-up utility gap
    [11]    utility-distribution entropy (same softmax-entropy formula as
            ``BoundaryAnalyzer.analyze``)
    [12:15] controller drift state: drift_ema, trust_radius, drift_integral
    [15]    violated-constraint-set transition indicator (Jaccard distance
            from the previous step; 0.0 at the first step)
    [16]    selected-model transition indicator (1.0 if the selected/
            recovered model differs from the previous step; 0.0 at the
            first step)

Not modeled in this controlled track: per-model utility *components*
(quality/latency/cost/...) and resource-utilization history, since the
controlled dataset generator (``dataset.py``) deliberately bypasses
``CompitumRouter`` and therefore has no ``SymbolicFreeEnergy`` breakdown or
real usage/cost telemetry to draw on. Both are candidates for a later
realized-routing track, not fabricated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..control import LyapunovController
from .static import ConstraintPressureResult

CHANNEL_DIMENSION = 17
SLACK_SCALE = 2.0  # matches the controlled dataset's own b magnitude (0..2)


@dataclass
class ChannelPreviousState:
    violated_indices: Set[int] = field(default_factory=set)
    selected_model: Optional[str] = None


def _utility_entropy(utilities_ordered: List[float]) -> float:
    """Identical formula to ``BoundaryAnalyzer.analyze``'s softmax entropy,
    recomputed here since the controlled track never calls that class."""
    arr = np.array(utilities_ordered, dtype=float)
    probs = np.exp(arr - arr.max())
    probs /= probs.sum()
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def _jaccard_distance(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    intersection = a & b
    return 1.0 - (len(intersection) / len(union))


def compute_channel_vector(
    pressure: ConstraintPressureResult,
    utilities: Dict[str, float],
    controller: LyapunovController,
    model_names: Sequence[str],
    previous: Optional[ChannelPreviousState],
) -> Tuple[np.ndarray, ChannelPreviousState]:
    """Compute one step's declared channel vector.

    ``controller`` must be a single ``LyapunovController`` instance reused
    across every step of a sequence, so its drift state genuinely evolves
    (the controller is mutated by this call, exactly as ``CompitumRouter``
    mutates its own controller during real routing). ``previous`` is
    ``None`` only for a sequence's first step.
    """
    slack = np.array([t.current_slack for t in pressure.targets], dtype=float)
    normalized_slack = np.clip(slack / SLACK_SCALE, -1.0, 1.0)

    violated_indices = {t.index for t in pressure.targets if t.already_violated}
    linear_feasible = not violated_indices
    feasibility_mask = []
    for name in model_names:
        if linear_feasible:
            # A model is truly feasible iff it is capable AND is the
            # case's actual selected model or shares its feasibility --
            # under linear feasibility, feasibility is capability alone;
            # the oracle doesn't expose per-model capability directly, so
            # we use the case-level signal: only the selected model (and
            # any tied-for-best) is confirmed feasible here, everything
            # else is unknown-but-not-selected. This mask therefore
            # answers "is this the (a) feasible pick", not a full per
            # -model capability probe -- documented, not silently assumed.
            feasibility_mask.append(1.0 if name == pressure.selected_model else 0.0)
        else:
            feasibility_mask.append(0.0)

    utilities_ordered = [utilities.get(name, 0.0) for name in model_names]
    ranked = sorted(utilities_ordered, reverse=True)
    utility_gap = (ranked[0] - ranked[1]) if len(ranked) > 1 else 0.0
    entropy = _utility_entropy(utilities_ordered) if utilities_ordered else 0.0

    d_star = max(0.0, float(-slack.min())) if len(slack) else 0.0
    _, drift_status = controller.update(d_star=d_star, grad_norm=1.0)

    if previous is None:
        violated_transition = 0.0
        model_transition = 0.0
    else:
        violated_transition = _jaccard_distance(violated_indices, previous.violated_indices)
        model_transition = (
            1.0
            if previous.selected_model is not None
            and previous.selected_model != pressure.selected_model
            else 0.0
        )

    vector = np.concatenate(
        [
            normalized_slack,
            np.array(feasibility_mask, dtype=float),
            np.array(utilities_ordered, dtype=float),
            np.array([utility_gap, entropy]),
            np.array(
                [
                    drift_status["drift_ema"],
                    drift_status["trust_radius"],
                    drift_status["drift_integral"],
                ]
            ),
            np.array([violated_transition, model_transition]),
        ]
    )
    assert vector.shape == (CHANNEL_DIMENSION,), vector.shape

    new_previous = ChannelPreviousState(
        violated_indices=set(violated_indices), selected_model=pressure.selected_model
    )
    return vector, new_previous


def compute_sequence_channels(
    pressures: List[ConstraintPressureResult],
    utilities_by_step: List[Dict[str, float]],
    model_names: Sequence[str],
    controller: Optional[LyapunovController] = None,
) -> List[np.ndarray]:
    """Compute the declared channel vector for every step of one sequence,
    with a single controller instance whose drift state evolves across the
    whole sequence."""
    if len(pressures) != len(utilities_by_step):
        raise ValueError("pressures and utilities_by_step must have equal length")
    ctrl = controller if controller is not None else LyapunovController()
    previous: Optional[ChannelPreviousState] = None
    vectors: List[np.ndarray] = []
    for pressure, utilities in zip(pressures, utilities_by_step):
        vector, previous = compute_channel_vector(pressure, utilities, ctrl, model_names, previous)
        vectors.append(vector)
    return vectors
