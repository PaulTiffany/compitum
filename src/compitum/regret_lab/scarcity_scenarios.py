"""Parameterized scarcity/opportunity-cost scenario generator (tranche 4.5).

Deliberately narrower than tranche 3's multi-resource environment: a
single resource ("budget") and three always-present models (``conserve``,
``spend``, ``opportunity``) so the scarcity/future-payoff relationship can
be read directly, without multi-resource-interaction confounds (tranche 3
already covers those separately). ``opportunity`` is declared in every
case (required so ``simulate_policy``'s per-sequence ``model_names`` stays
fixed) but is priced to be unconditionally infeasible except during its
declared window(s), where it becomes ``payoff_ratio x spend_utility`` at a
calibrated one-off cost -- modeling a rare, high-value event without a
separate per-case model set.

See docs/adr/0005-scarcity-response-study.md for the full design rationale
and the primary-grid/secondary-sweep split.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np

from .environment import DynamicCase, DynamicSequence, _grid, _stable_hash

SCARCITY_RESOURCE_NAMES: Tuple[str, ...] = ("budget",)
SCARCITY_MODEL_NAMES: Tuple[str, str, str] = ("conserve", "spend", "opportunity")

CONSERVE_UTILITY = 1.0
SPEND_UTILITY = 2.0
CONSERVE_RATE = 1.0
OPPORTUNITY_COST = 5.0
INFEASIBLE_CONSUMPTION = 1.0e6
STEPS = 12
PRIMARY_SEEDS_PER_CELL = 3
SECONDARY_SEEDS_PER_CELL = 5

PAYOFF_RATIOS: Tuple[float, ...] = (1.0, 1.5, 3.0, 10.0)
BUDGET_TIGHTNESS_LEVELS: Tuple[float, ...] = (2.0, 1.1, 1.0)
PRIMARY_REPLENISHMENT_MODES: Tuple[str, ...] = ("none", "partial")
TIMINGS: Tuple[str, ...] = ("near", "mid", "final")

CONSUMPTION_ASYMMETRY_LEVELS: Tuple[float, ...] = (1.2, 2.0, 4.0)
FORECAST_ERROR_MODES: Tuple[str, ...] = ("none", "over", "under", "delayed")
OPPORTUNITY_PREVALENCE_LEVELS: Tuple[str, ...] = ("rare", "moderate", "stochastic")
SECONDARY_REPLENISHMENT_MODES: Tuple[str, ...] = ("periodic", "delayed")

STOCHASTIC_BONUS_PROBABILITY = 0.15


@dataclass(frozen=True)
class ScarcityParams:
    payoff_ratio: float
    budget_tightness: float
    replenishment_mode: str
    timing: str
    consumption_asymmetry: float = 2.0
    forecast_error_mode: str = "none"
    opportunity_prevalence: str = "rare"

    def cell_id(self) -> str:
        return (
            f"pr{self.payoff_ratio}-bt{self.budget_tightness}-rep{self.replenishment_mode}-"
            f"t{self.timing}-ca{self.consumption_asymmetry}-fe{self.forecast_error_mode}-"
            f"op{self.opportunity_prevalence}"
        )


REFERENCE_PARAMS = ScarcityParams(
    payoff_ratio=3.0, budget_tightness=1.1, replenishment_mode="none", timing="final"
)


def _timing_step(timing: str, steps: int) -> int:
    if timing == "near":
        return 1
    if timing == "mid":
        return steps // 2
    if timing == "final":
        return steps - 1
    raise ValueError(f"unknown timing {timing!r}")


def _replenishment_schedule(mode: str, steps: int) -> List[float]:
    if mode == "none":
        return [0.0] * steps
    if mode == "partial":
        return [0.5] * steps
    if mode == "periodic":
        period_count = steps // 4
        total = 0.5 * steps
        per_period = total / period_count if period_count else 0.0
        return [per_period if (t + 1) % 4 == 0 else 0.0 for t in range(steps)]
    if mode == "delayed":
        half = steps // 2
        return [0.0 if t < half else 1.0 for t in range(steps)]
    raise ValueError(f"unknown replenishment_mode {mode!r}")


def _secondary_windows(
    rng: np.random.Generator, params: ScarcityParams, steps: int, t_opp: int
) -> Dict[int, float]:
    windows: Dict[int, float] = {}
    if params.opportunity_prevalence == "moderate":
        for candidate in (steps // 4, (3 * steps) // 4):
            if candidate != t_opp and 0 <= candidate < steps:
                windows[candidate] = params.payoff_ratio / 2.0
    elif params.opportunity_prevalence == "stochastic":
        for t in range(steps):
            if t != t_opp and rng.random() < STOCHASTIC_BONUS_PROBABILITY:
                windows[t] = params.payoff_ratio / 2.0
    return windows


def build_scarcity_sequence(
    rng: np.random.Generator,
    params: ScarcityParams,
    steps: int,
    sequence_id: str,
    tightness_reference_rate: float = CONSERVE_RATE,
) -> DynamicSequence:
    """``tightness_reference_rate`` is the per-step consumption rate
    ``budget_tightness`` is calibrated against. The default, ``CONSERVE_RATE``,
    matches tranche 4.5's original cells exactly (calibrated against a fully
    -conservative reference). Tranche 4.6 found this understates true slack
    at short horizons, since a naturally-behaving (spend-preferring) policy
    consumes at ``params.consumption_asymmetry``'s rate, not ``CONSERVE_RATE``
    -- passing ``tightness_reference_rate=params.consumption_asymmetry``
    calibrates against that natural reference instead. See ADR 0005's
    tranche-4.6 addendum."""
    t_opp = _timing_step(params.timing, steps)
    min_budget_needed = t_opp * tightness_reference_rate + OPPORTUNITY_COST
    initial_budget = {"budget": _grid(params.budget_tightness * min_budget_needed)}
    replenishment_schedule = _replenishment_schedule(params.replenishment_mode, steps)
    secondary_windows = _secondary_windows(rng, params, steps, t_opp)

    cases = []
    for t in range(steps):
        utility = {"conserve": CONSERVE_UTILITY, "spend": SPEND_UTILITY, "opportunity": 0.0}
        expected = {
            "conserve": {"budget": CONSERVE_RATE},
            "spend": {"budget": params.consumption_asymmetry},
            "opportunity": {"budget": INFEASIBLE_CONSUMPTION},
        }
        realized = {
            "conserve": {"budget": CONSERVE_RATE},
            "spend": {"budget": params.consumption_asymmetry},
            "opportunity": {"budget": INFEASIBLE_CONSUMPTION},
        }
        revelation_delay = 0

        window_ratio = None
        if t == t_opp:
            window_ratio = params.payoff_ratio
        elif t in secondary_windows:
            window_ratio = secondary_windows[t]

        if window_ratio is not None:
            utility["opportunity"] = window_ratio * SPEND_UTILITY
            true_cost = OPPORTUNITY_COST if t == t_opp else OPPORTUNITY_COST / 2.0
            expected_cost = true_cost
            if params.forecast_error_mode == "over":
                expected_cost = true_cost * 1.3
            elif params.forecast_error_mode == "under":
                expected_cost = true_cost * 0.7
            expected["opportunity"] = {"budget": expected_cost}
            realized["opportunity"] = {"budget": true_cost}
            if params.forecast_error_mode == "delayed" and t == t_opp:
                revelation_delay = 3

        cases.append(
            DynamicCase(
                step=t,
                base_utility=utility,
                expected_consumption={
                    m: {r: _grid(v) for r, v in d.items()} for m, d in expected.items()
                },
                realized_consumption={
                    m: {r: _grid(v) for r, v in d.items()} for m, d in realized.items()
                },
                revelation_delay=revelation_delay,
                replenishment={"budget": _grid(replenishment_schedule[t])},
            )
        )

    return DynamicSequence(
        sequence_id=sequence_id,
        scenario=params.cell_id(),
        resource_names=SCARCITY_RESOURCE_NAMES,
        model_names=SCARCITY_MODEL_NAMES,
        initial_budget=initial_budget,
        cases=cases,
    )


def primary_grid() -> List[ScarcityParams]:
    return [
        ScarcityParams(payoff_ratio=pr, budget_tightness=bt, replenishment_mode=rep, timing=t)
        for pr in PAYOFF_RATIOS
        for bt in BUDGET_TIGHTNESS_LEVELS
        for rep in PRIMARY_REPLENISHMENT_MODES
        for t in TIMINGS
    ]


def secondary_sweeps() -> Dict[str, List[ScarcityParams]]:
    return {
        "consumption_asymmetry": [
            replace(REFERENCE_PARAMS, consumption_asymmetry=v) for v in CONSUMPTION_ASYMMETRY_LEVELS
        ],
        "forecast_error_mode": [
            replace(REFERENCE_PARAMS, forecast_error_mode=v) for v in FORECAST_ERROR_MODES
        ],
        "opportunity_prevalence": [
            replace(REFERENCE_PARAMS, opportunity_prevalence=v)
            for v in OPPORTUNITY_PREVALENCE_LEVELS
        ],
        "replenishment_mode": [
            replace(REFERENCE_PARAMS, replenishment_mode=v) for v in SECONDARY_REPLENISHMENT_MODES
        ],
    }


def generate_primary_dataset(
    seed: int, steps: int = STEPS, seeds_per_cell: int = PRIMARY_SEEDS_PER_CELL
) -> List[DynamicSequence]:
    out = []
    for params in primary_grid():
        for index in range(seeds_per_cell):
            rng = np.random.default_rng((seed, _stable_hash(params.cell_id()), index))
            seq_id = f"{params.cell_id()}-{index:03d}"
            out.append(build_scarcity_sequence(rng, params, steps, seq_id))
    return out


def generate_corrected_slack_dataset(
    seed: int, steps: int = STEPS, seeds_per_cell: int = PRIMARY_SEEDS_PER_CELL
) -> List[DynamicSequence]:
    """Tranche 4.6: regenerates the near/mid-timing slice of the primary
    grid (all payoff_ratio x budget_tightness x replenishment_mode
    combinations, 24 cells) calibrated against the natural
    spend-preferring reference rate instead of the fully-conservative one.
    Sequence ids are prefixed ``corrected-`` so they never collide with
    ``generate_primary_dataset``'s original cells; both artifacts are kept,
    never merged or used to overwrite each other."""
    out = []
    for params in primary_grid():
        if params.timing not in ("near", "mid"):
            continue
        for index in range(seeds_per_cell):
            rng = np.random.default_rng((seed, _stable_hash(params.cell_id()), index))
            seq_id = f"corrected-{params.cell_id()}-{index:03d}"
            out.append(
                build_scarcity_sequence(
                    rng,
                    params,
                    steps,
                    seq_id,
                    tightness_reference_rate=params.consumption_asymmetry,
                )
            )
    return out


def generate_secondary_dataset(
    seed: int, steps: int = STEPS, seeds_per_cell: int = SECONDARY_SEEDS_PER_CELL
) -> Dict[str, List[DynamicSequence]]:
    out: Dict[str, List[DynamicSequence]] = {}
    for axis, params_list in secondary_sweeps().items():
        sequences = []
        for params in params_list:
            for index in range(seeds_per_cell):
                seed_key = f"{axis}:{params.cell_id()}"
                rng = np.random.default_rng((seed, _stable_hash(seed_key), index))
                seq_id = f"{axis}-{params.cell_id()}-{index:03d}"
                sequences.append(build_scarcity_sequence(rng, params, steps, seq_id))
        out[axis] = sequences
    return out
