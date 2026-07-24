"""Experiment-owned dynamic resource/consumption environment (tranche 3).

Declares route-specific, time-varying resource consumption
(``consumption[t, model, resource]``) and cumulative/replenishing resource
state across a sequence, per docs/adr/0003-dynamic-constraint-regret.md.
This is a new, independent environment -- it does not read or modify
``compitum.constraints`` or ``ReflectiveConstraintSolver`` in any way, since
the frozen solver has no notion of cumulative, route-specific resource
state (that is precisely the gap tranche 2 found).

All consumption/capacity/replenishment quantities are generated as exact
multiples of ``GRID_UNIT`` so the hindsight oracle's integer-scaled DP
(``hindsight.py``) never suffers floating-point drift.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

GRID_UNIT = 0.25
RESOURCE_NAMES: Tuple[str, str] = ("budget", "quota")
MODEL_NAMES: Tuple[str, str, str] = ("economy", "standard", "premium")

BASE_UTILITY: Dict[str, float] = {"economy": 1.0, "standard": 2.0, "premium": 3.5}
BASE_CONSUMPTION: Dict[str, Dict[str, float]] = {
    "economy": {"budget": 0.25, "quota": 0.25},
    "standard": {"budget": 0.75, "quota": 0.50},
    "premium": {"budget": 1.50, "quota": 1.25},
}

SCENARIOS: Tuple[str, ...] = (
    "permanently_slack",
    "single_resource_scarce_period",
    "demand_burst",
    "conserve_enables_better_future",
    "premature_conservation_regret",
    "multi_resource_interaction",
    "forecast_error",
    "delayed_realization",
)


def _grid(x: float) -> float:
    return round(round(x / GRID_UNIT) * GRID_UNIT, 10)


def _stable_hash(text: str) -> int:
    """Process-stable string hash for RNG seeding. Python's built-in
    ``hash()`` is randomized per process (PEP 456 SipHash salt) unless
    ``PYTHONHASHSEED`` is fixed, which would make ``generate_dynamic_dataset``
    only reproducible *within* one process, not across separate runs of the
    same ``seed`` -- exactly the failure mode this function avoids."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _jitter_utility(rng: np.random.Generator) -> Dict[str, float]:
    return {m: round(BASE_UTILITY[m] + float(rng.uniform(-0.05, 0.05)), 4) for m in MODEL_NAMES}


def _flat_consumption() -> Dict[str, Dict[str, float]]:
    return {m: dict(BASE_CONSUMPTION[m]) for m in MODEL_NAMES}


@dataclass
class DynamicCase:
    step: int
    base_utility: Dict[str, float]
    expected_consumption: Dict[str, Dict[str, float]]
    realized_consumption: Dict[str, Dict[str, float]]
    revelation_delay: int
    replenishment: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "base_utility": dict(self.base_utility),
            "expected_consumption": {m: dict(v) for m, v in self.expected_consumption.items()},
            "realized_consumption": {m: dict(v) for m, v in self.realized_consumption.items()},
            "revelation_delay": self.revelation_delay,
            "replenishment": dict(self.replenishment),
        }


@dataclass
class DynamicSequence:
    sequence_id: str
    scenario: str
    resource_names: Tuple[str, ...]
    model_names: Tuple[str, ...]
    initial_budget: Dict[str, float]
    cases: List[DynamicCase]


def _case(
    step: int,
    base_utility: Dict[str, float],
    expected: Dict[str, Dict[str, float]],
    realized: Dict[str, Dict[str, float]],
    replenishment: Dict[str, float],
    revelation_delay: int = 0,
) -> DynamicCase:
    return DynamicCase(
        step=step,
        base_utility=base_utility,
        expected_consumption={m: {r: _grid(v) for r, v in d.items()} for m, d in expected.items()},
        realized_consumption={m: {r: _grid(v) for r, v in d.items()} for m, d in realized.items()},
        revelation_delay=revelation_delay,
        replenishment={r: _grid(v) for r, v in replenishment.items()},
    )


def _steady_replenishment() -> Dict[str, float]:
    """Replenishment matching baseline steady-state usage of the 'standard'
    model -- enough that a sequence choosing 'standard' every step neither
    grows nor drains its budget, a neutral reference point for scenarios
    that otherwise keep resources ample."""
    return dict(BASE_CONSUMPTION["standard"])


def _scenario_permanently_slack(rng: np.random.Generator, steps: int) -> DynamicSequence:
    initial = {"budget": steps * 3.0, "quota": steps * 3.0}
    cases = [
        _case(
            t,
            _jitter_utility(rng),
            _flat_consumption(),
            _flat_consumption(),
            _steady_replenishment(),
        )
        for t in range(steps)
    ]
    return _sequence("permanently_slack", initial, cases)


def _scenario_single_resource_scarce_period(
    rng: np.random.Generator, steps: int
) -> DynamicSequence:
    window = range(steps // 3, 2 * steps // 3)
    initial = {"budget": steps * 3.0, "quota": 2.0}
    cases = []
    for t in range(steps):
        utility = _jitter_utility(rng)
        utility["premium"] += 3.0  # always clearly best -- feasibility is the only question
        replen = {"budget": 1.0, "quota": 0.0 if t in window else 0.5}
        cases.append(_case(t, utility, _flat_consumption(), _flat_consumption(), replen))
    return _sequence("single_resource_scarce_period", initial, cases)


def _scenario_demand_burst(rng: np.random.Generator, steps: int) -> DynamicSequence:
    burst_step = steps - steps // 4
    initial = {"budget": steps * 1.2, "quota": steps * 1.1}
    cases = []
    for t in range(steps):
        utility = _jitter_utility(rng)
        if t == burst_step:
            utility["premium"] += 15.0
        replen = {"budget": 0.6, "quota": 0.55}
        cases.append(_case(t, utility, _flat_consumption(), _flat_consumption(), replen))
    return _sequence("demand_burst", initial, cases)


def _scenario_conserve_enables_better_future(
    rng: np.random.Generator, steps: int
) -> DynamicSequence:
    initial = {"budget": steps * 0.9, "quota": steps * 3.0}
    cases = []
    for t in range(steps):
        utility = _jitter_utility(rng)
        if t == steps - 1:
            utility["premium"] += 20.0  # one large, un-repeatable payoff at the very end
        replen = {"budget": 0.0, "quota": 1.0}  # budget never replenishes: permanently draining
        cases.append(_case(t, utility, _flat_consumption(), _flat_consumption(), replen))
    return _sequence("conserve_enables_better_future", initial, cases)


def _scenario_premature_conservation_regret(
    rng: np.random.Generator, steps: int
) -> DynamicSequence:
    spike_step = 1
    initial = {"budget": steps * 3.0, "quota": steps * 3.0}
    cases = []
    for t in range(steps):
        utility = _jitter_utility(rng)
        expected = _flat_consumption()
        realized = _flat_consumption()
        if t == spike_step:
            # A one-off consumption spike on whichever model gets picked,
            # immediately followed by ample replenishment: resources are
            # never genuinely scarce again, but a controller that overreacts
            # to this transient utilization error and is slow to relax will
            # under-use high-utility models afterward for no true reason.
            for m in MODEL_NAMES:
                realized[m] = {r: v * 3.0 for r, v in realized[m].items()}
        replen = {"budget": 2.0, "quota": 2.0}
        cases.append(_case(t, utility, expected, realized, replen))
    return _sequence("premature_conservation_regret", initial, cases)


def _scenario_multi_resource_interaction(rng: np.random.Generator, steps: int) -> DynamicSequence:
    half = steps // 2
    initial = {"budget": steps * 1.0, "quota": steps * 1.0}
    cases = []
    for t in range(steps):
        utility = _jitter_utility(rng)
        # First half: quota is the tight resource (low replenishment);
        # second half: budget is the tight resource instead. Premium is
        # heavy on budget, economy is heavy on quota, so the jointly
        # -optimal path has to balance both constraints over time, not
        # just watch a single resource.
        if t < half:
            replen = {"budget": 1.0, "quota": 0.25}
        else:
            replen = {"budget": 0.25, "quota": 1.0}
        cases.append(_case(t, utility, _flat_consumption(), _flat_consumption(), replen))
    return _sequence("multi_resource_interaction", initial, cases)


def _scenario_forecast_error(rng: np.random.Generator, steps: int) -> DynamicSequence:
    initial = {"budget": steps * 1.5, "quota": steps * 1.5}
    cases = []
    for t in range(steps):
        utility = _jitter_utility(rng)
        expected = _flat_consumption()
        realized = _flat_consumption()
        # Premium's true budget draw is systematically understated by its
        # own forecast -- a fixed bias plus noise, so a policy that trusts
        # expected_consumption at face value will be surprised, while one
        # that learns the residual (the EWMA forecaster) will not.
        bias = 0.5 + float(rng.uniform(-0.1, 0.1))
        realized["premium"]["budget"] = expected["premium"]["budget"] + bias
        replen = {"budget": 1.0, "quota": 1.0}
        cases.append(_case(t, utility, expected, realized, replen))
    return _sequence("forecast_error", initial, cases)


def _scenario_delayed_realization(rng: np.random.Generator, steps: int) -> DynamicSequence:
    initial = {"budget": steps * 1.0, "quota": steps * 1.0}
    cases = []
    for t in range(steps):
        utility = _jitter_utility(rng)
        expected = _flat_consumption()
        realized = _flat_consumption()
        for m in MODEL_NAMES:
            for r in RESOURCE_NAMES:
                realized[m][r] = expected[m][r] + float(rng.uniform(-0.15, 0.35))
        replen = {"budget": 0.5, "quota": 0.5}
        cases.append(_case(t, utility, expected, realized, replen, revelation_delay=2))
    return _sequence("delayed_realization", initial, cases)


_BUILDERS = {
    "permanently_slack": _scenario_permanently_slack,
    "single_resource_scarce_period": _scenario_single_resource_scarce_period,
    "demand_burst": _scenario_demand_burst,
    "conserve_enables_better_future": _scenario_conserve_enables_better_future,
    "premature_conservation_regret": _scenario_premature_conservation_regret,
    "multi_resource_interaction": _scenario_multi_resource_interaction,
    "forecast_error": _scenario_forecast_error,
    "delayed_realization": _scenario_delayed_realization,
}


def _sequence(
    scenario: str, initial: Dict[str, float], cases: List[DynamicCase]
) -> DynamicSequence:
    return DynamicSequence(
        sequence_id="",
        scenario=scenario,
        resource_names=RESOURCE_NAMES,
        model_names=MODEL_NAMES,
        initial_budget={r: _grid(v) for r, v in initial.items()},
        cases=cases,
    )


def generate_dynamic_dataset(
    seed: int,
    sequences_per_scenario: int = 4,
    steps_per_sequence: int = 10,
    scenarios: Tuple[str, ...] = SCENARIOS,
) -> List[DynamicSequence]:
    """Deterministic, reproducible dataset generator: one independently
    seeded RNG stream per (scenario, index) so any single sequence can be
    regenerated in isolation."""
    out: List[DynamicSequence] = []
    for scenario in scenarios:
        builder = _BUILDERS[scenario]
        for index in range(sequences_per_scenario):
            rng = np.random.default_rng((seed, _stable_hash(scenario), index))
            seq = builder(rng, steps_per_sequence)
            seq.sequence_id = f"{scenario}-{index:03d}"
            out.append(seq)
    return out
