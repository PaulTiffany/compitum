"""Controlled constraint-pressure dataset generator (tranche 2, track 1).

Deterministic, reproducible routing SEQUENCES built directly on Compitum's
own ``Model``/``Capabilities``/``ReflectiveConstraintSolver`` (imported
unmodified, never stubbed), with independently-controlled slack, utility
gap, suppressed-competitor identity, and capability eligibility.

This deliberately does NOT go through ``CompitumRouter.route()``: that
method hardcodes ``xB = np.zeros(4)`` whenever an embedding is supplied (see
``src/compitum/router.py``), which would make every case trivially deep
-feasible and useless for a constraint-pressure study. Cases here construct
``(xB, utilities, context)`` directly instead, per the explicit tranche 2
sequencing: validate the oracle and this controlled track before touching
any realized routing dataset.

Each sequence uses the project's own default constraint config shape (a
4-row diagonal ``A`` over ``b = [2.0, 2.0, 0.0, 0.0]``, matching
``configs/constraints_us_default.yaml``) so the controlled track and any
later real-config track share the same constraint semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..capabilities import Capabilities
from ..models import Model

DEFAULT_A = np.eye(4)
DEFAULT_B = np.array([2.0, 2.0, 0.0, 0.0])
CONSTRAINT_NAMES = ("latency_class", "cost_class", "pii_level", "region_eu_only")

SCENARIOS = (
    "permanently_slack",
    "single_constraint_ramp_recoverable",
    "single_constraint_ramp_capability_blocked",
    "single_constraint_ramp_no_higher_utility",
    "single_constraint_ramp_all_capability_blocked",
    "multi_constraint_joint",
    "discontinuous_tie",
    "unbinding_recovery",
    "permanently_infeasible",
)


@dataclass
class ControlledCase:
    step: int
    xB: np.ndarray
    utilities: Dict[str, float]
    context: Optional[Dict[str, Any]]


@dataclass
class ControlledSequence:
    sequence_id: str
    scenario: str
    A: np.ndarray
    b: np.ndarray
    models: List[Model]
    cases: List[ControlledCase] = field(default_factory=list)
    known_binding_step: Optional[int] = None
    known_binding_index: Optional[int] = None


def _models(rng: np.random.Generator, suppressed_capable: bool) -> List[Model]:
    """Three models: 'current' (always capable, moderate utility), 'suppressed'
    (higher utility, capability controlled by the caller), and 'filler' (a
    third, low-utility distractor, always capable)."""
    regions = {"US", "CA", "EU"} if suppressed_capable else {"EU"}
    return [
        Model(
            name="current",
            center=rng.normal(size=2),
            capabilities=Capabilities(regions={"US", "CA", "EU"}, tools_allowed={"none"}),
            cost=0.1,
        ),
        Model(
            name="suppressed",
            center=rng.normal(size=2),
            capabilities=Capabilities(regions=regions, tools_allowed={"none"}),
            cost=0.1,
        ),
        Model(
            name="filler",
            center=rng.normal(size=2),
            capabilities=Capabilities(regions={"US", "CA", "EU"}, tools_allowed={"none"}),
            cost=0.1,
        ),
    ]


def _ramp(start: float, end: float, steps: int) -> np.ndarray:
    if steps == 1:
        return np.array([start])
    return np.linspace(start, end, steps)


def _binding_step(slack_series: np.ndarray, epsilon: float = 1e-10) -> Optional[int]:
    below = np.nonzero(slack_series < -epsilon)[0]
    return int(below[0]) if len(below) else None


def _sequence_from_slack_series(
    sequence_id: str,
    scenario: str,
    constraint_index: int,
    slack_series: np.ndarray,
    models: List[Model],
    utility_gap: float,
    context: Optional[Dict[str, Any]],
) -> ControlledSequence:
    xB_series = []
    for slack in slack_series:
        xB = np.zeros(4)
        xB[constraint_index] = DEFAULT_B[constraint_index] - slack
        xB_series.append(xB)
    utilities = {"current": 1.0, "suppressed": 1.0 + utility_gap, "filler": 0.1}
    cases = [
        ControlledCase(step=t, xB=xb, utilities=utilities, context=context)
        for t, xb in enumerate(xB_series)
    ]
    return ControlledSequence(
        sequence_id=sequence_id,
        scenario=scenario,
        A=DEFAULT_A,
        b=DEFAULT_B,
        models=models,
        cases=cases,
        known_binding_step=_binding_step(slack_series),
        known_binding_index=constraint_index,
    )


def _generate_one(
    rng: np.random.Generator, scenario: str, index: int, steps: int
) -> ControlledSequence:
    sequence_id = f"{scenario}-{index:03d}"
    constraint_index = int(rng.integers(0, 4))
    utility_gap = float(rng.uniform(0.5, 5.0))

    if scenario == "permanently_slack":
        models = _models(rng, suppressed_capable=True)
        slack_series = np.full(steps, 1.5)  # xB deep inside bounds throughout
        return _sequence_from_slack_series(
            sequence_id, scenario, constraint_index, slack_series, models, utility_gap, None
        )

    if scenario == "single_constraint_ramp_recoverable":
        models = _models(rng, suppressed_capable=True)
        slack_series = _ramp(1.5, -1.0, steps)
        return _sequence_from_slack_series(
            sequence_id, scenario, constraint_index, slack_series, models, utility_gap, None
        )

    if scenario == "single_constraint_ramp_capability_blocked":
        # "suppressed" (the deliberately high-utility competitor) is region
        # -blocked, but "current"/"filler" remain fully capable: relaxing the
        # sole violated constraint still recovers feasibility via one of
        # them (any current violation blocks EVERY model, including
        # "current" itself), it just never recovers via "suppressed"
        # specifically.
        models = _models(rng, suppressed_capable=False)
        slack_series = _ramp(1.5, -1.0, steps)
        context = {"region": "US"}  # "suppressed" only supports EU
        return _sequence_from_slack_series(
            sequence_id, scenario, constraint_index, slack_series, models, utility_gap, context
        )

    if scenario == "single_constraint_ramp_all_capability_blocked":
        # Every model in the pool is region-blocked: relaxing the sole
        # violated constraint can never recover ANY pick, regardless of
        # utility -- the genuine "capability_blocked_only" case.
        rng2 = np.random.default_rng(rng.integers(0, 2**31))
        blocked = {"EU"}
        models = [
            Model(
                name="current",
                center=rng2.normal(size=2),
                capabilities=Capabilities(regions=blocked, tools_allowed={"none"}),
                cost=0.1,
            ),
            Model(
                name="suppressed",
                center=rng2.normal(size=2),
                capabilities=Capabilities(regions=blocked, tools_allowed={"none"}),
                cost=0.1,
            ),
            Model(
                name="filler",
                center=rng2.normal(size=2),
                capabilities=Capabilities(regions=blocked, tools_allowed={"none"}),
                cost=0.1,
            ),
        ]
        slack_series = _ramp(1.5, -1.0, steps)
        context = {"region": "US"}
        return _sequence_from_slack_series(
            sequence_id, scenario, constraint_index, slack_series, models, utility_gap, context
        )

    if scenario == "single_constraint_ramp_no_higher_utility":
        # "suppressed" has deliberately LOW utility relative to "current".
        # This distinction is only meaningful in the linear-feasible branch
        # (where relaxation is compared against an established m_star); in
        # the infeasible-fallback branch there is no such baseline, so
        # relaxing the sole violated constraint still recovers "current"
        # itself (the highest-utility capable model) -- it just never
        # recovers via "suppressed".
        models = _models(rng, suppressed_capable=True)
        slack_series = _ramp(1.5, -1.0, steps)
        seq = _sequence_from_slack_series(
            sequence_id, scenario, constraint_index, slack_series, models, utility_gap, None
        )
        for case in seq.cases:
            case.utilities = {"current": 5.0, "suppressed": 1.0, "filler": 0.1}
        return seq

    if scenario == "multi_constraint_joint":
        models = _models(rng, suppressed_capable=True)
        other_index = (constraint_index + 1) % 4
        slack_a = _ramp(1.5, -1.0, steps)
        slack_b = _ramp(1.5, -1.0, steps)
        xB_series = []
        for a, b in zip(slack_a, slack_b):
            xB = np.zeros(4)
            xB[constraint_index] = DEFAULT_B[constraint_index] - a
            xB[other_index] = DEFAULT_B[other_index] - b
            xB_series.append(xB)
        utilities = {"current": 1.0, "suppressed": 1.0 + utility_gap, "filler": 0.1}
        cases = [
            ControlledCase(step=t, xB=xb, utilities=utilities, context=None)
            for t, xb in enumerate(xB_series)
        ]
        return ControlledSequence(
            sequence_id=sequence_id,
            scenario=scenario,
            A=DEFAULT_A,
            b=DEFAULT_B,
            models=models,
            cases=cases,
            known_binding_step=_binding_step(slack_a),
            known_binding_index=constraint_index,
        )

    if scenario == "discontinuous_tie":
        rng2 = np.random.default_rng(rng.integers(0, 2**31))
        models = [
            Model(
                name="current",
                center=rng2.normal(size=2),
                capabilities=Capabilities(regions={"US", "CA", "EU"}, tools_allowed={"none"}),
                cost=0.1,
            ),
            Model(
                name="tie_a",
                center=rng2.normal(size=2),
                capabilities=Capabilities(regions={"US", "CA", "EU"}, tools_allowed={"none"}),
                cost=0.1,
            ),
            Model(
                name="tie_b",
                center=rng2.normal(size=2),
                capabilities=Capabilities(regions={"US", "CA", "EU"}, tools_allowed={"none"}),
                cost=0.1,
            ),
        ]
        slack_series = _ramp(1.5, -1.0, steps)
        xB_series = []
        for slack in slack_series:
            xB = np.zeros(4)
            xB[constraint_index] = DEFAULT_B[constraint_index] - slack
            xB_series.append(xB)
        utilities = {
            "current": 1.0,
            "tie_a": 1.0 + utility_gap,
            "tie_b": 1.0 + utility_gap,
        }
        cases = [
            ControlledCase(step=t, xB=xb, utilities=utilities, context=None)
            for t, xb in enumerate(xB_series)
        ]
        return ControlledSequence(
            sequence_id=sequence_id,
            scenario=scenario,
            A=DEFAULT_A,
            b=DEFAULT_B,
            models=models,
            cases=cases,
            known_binding_step=_binding_step(slack_series),
            known_binding_index=constraint_index,
        )

    if scenario == "unbinding_recovery":
        models = _models(rng, suppressed_capable=True)
        slack_series = _ramp(-1.0, 1.5, steps)  # starts violated, recovers
        return _sequence_from_slack_series(
            sequence_id, scenario, constraint_index, slack_series, models, utility_gap, None
        )

    if scenario == "permanently_infeasible":
        models = _models(rng, suppressed_capable=True)
        slack_series = np.full(steps, -1.0)
        return _sequence_from_slack_series(
            sequence_id, scenario, constraint_index, slack_series, models, utility_gap, None
        )

    raise ValueError(f"unknown scenario: {scenario!r}")


def generate_controlled_dataset(
    seed: int,
    sequences_per_scenario: int = 6,
    steps_per_sequence: int = 8,
    scenarios: Optional[List[str]] = None,
) -> List[ControlledSequence]:
    """Generate the controlled constraint-pressure track.

    Deterministic given ``seed``: reruns with the same seed produce
    byte-identical ``xB``/utility values (verified in
    ``tests/constraint_oracle/test_dataset.py``).
    """
    if steps_per_sequence < 1:
        raise ValueError("steps_per_sequence must be >= 1")
    chosen = scenarios if scenarios is not None else list(SCENARIOS)
    unknown = set(chosen) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"unknown scenario(s): {sorted(unknown)}")

    sequences: List[ControlledSequence] = []
    rng = np.random.default_rng(seed)
    for scenario in chosen:
        for index in range(sequences_per_scenario):
            sequences.append(_generate_one(rng, scenario, index, steps_per_sequence))
    return sequences
