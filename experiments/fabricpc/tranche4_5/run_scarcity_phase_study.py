"""Tranche 4.5 frozen-controller scarcity phase-diagram study.

Pure ``compitum.regret_lab`` -- no FabricPC, no JAX, no parameter tuning.
Per docs/adr/0005-scarcity-response-study.md: tranche 4's frozen pacing
controller cut mean regret 8x, but the whole effect was 2 of 16 sequences
(both `conserve_enables_better_future`). This script freezes tranche 4's
selected pacing-family parameters exactly (no retuning per cell) and
varies the ENVIRONMENT across a declared scarcity phase space to determine
whether the earlier result reflects a coherent, generalizable relationship
or an artifact of one extreme construction.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from compitum.regret_lab import (
    DynamicSequence,
    HindsightResult,
    PacingController,
    PolicyRunResult,
    ReactiveController,
    bootstrap_ci,
    compute_hindsight_optimum,
    conservation_depletion_split,
    generate_primary_dataset,
    generate_secondary_dataset,
    paired_regret_deltas,
    primary_grid,
    regret_metrics,
    simulate_policy,
    total_available_over_horizon,
)
from compitum.regret_lab.scarcity_scenarios import ScarcityParams
from compitum.regret_lab.simulator import PolicyDecision

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche4_5" / "artifacts"
SEED = 4242

# Frozen tranche 4 configurations -- read directly from
# experiments/fabricpc/tranche4/artifacts/pilot_report.json's
# parameter_selection section. NOT retuned anywhere in this script.
DUAL_ETA = 0.5
DUAL_LAMBDA_MAX = 20.0
FROZEN_PACING: Dict[str, float] = {"eta": 1.8}
FROZEN_HYSTERESIS: Dict[str, float] = {"eta": 2.0, "deadband": 0.05}
FROZEN_ASYMMETRIC: Dict[str, float] = {"eta": 1.5, "relax_gamma": 2.0, "rise_scale": 1.0}
FROZEN_BOUNDED_SMOOTHED: Dict[str, float] = {"eta": 2.0, "ema_alpha": 0.5, "max_step": 1.0}

# Preregistered thresholds (ADR 0005) -- declared before any evaluation.
H1_REGRET_TOLERANCE = 0.05
H1_DISAGREEMENT_TOLERANCE = 0.05
H5_UNUSED_RESOURCE_TOLERANCE = 0.5
MEANINGFUL_IMPROVEMENT_THRESHOLD = -0.1
GATE4_NON_INFERIORITY_TOLERANCE = 0.1
MIN_DISTINCT_CONFIGS_FOR_H2 = 2
MIN_SEQUENCES_FOR_GATE5 = 6
MAX_BOUNDARY_JUMP = 5.0

Factory = Callable[[DynamicSequence], Optional[Any]]


def _no_pricing(seq: DynamicSequence) -> None:
    return None


def _reactive(seq: DynamicSequence) -> ReactiveController:
    return ReactiveController(
        resource_names=seq.resource_names, eta=DUAL_ETA, lambda_max=DUAL_LAMBDA_MAX
    )


def _pacing(seq: DynamicSequence) -> PacingController:
    return PacingController(
        resource_names=seq.resource_names,
        total_available=total_available_over_horizon(seq),
        total_steps=len(seq.cases),
        **FROZEN_PACING,
    )


def _pacing_hysteresis(seq: DynamicSequence) -> PacingController:
    return PacingController(
        resource_names=seq.resource_names,
        total_available=total_available_over_horizon(seq),
        total_steps=len(seq.cases),
        **FROZEN_HYSTERESIS,
    )


def _pacing_asymmetric(seq: DynamicSequence) -> PacingController:
    return PacingController(
        resource_names=seq.resource_names,
        total_available=total_available_over_horizon(seq),
        total_steps=len(seq.cases),
        **FROZEN_ASYMMETRIC,
    )


def _pacing_bounded_smoothed(seq: DynamicSequence) -> PacingController:
    return PacingController(
        resource_names=seq.resource_names,
        total_available=total_available_over_horizon(seq),
        total_steps=len(seq.cases),
        **FROZEN_BOUNDED_SMOOTHED,
    )


ARMS: Dict[str, Factory] = {
    "no_pricing": _no_pricing,
    "reactive": _reactive,
    "pacing": _pacing,
    "pacing_hysteresis": _pacing_hysteresis,
    "pacing_asymmetric": _pacing_asymmetric,
    "pacing_bounded_smoothed": _pacing_bounded_smoothed,
}
PRIMARY_ARM = "pacing"


RunResult = Dict[str, Tuple[PolicyRunResult, List[PolicyDecision]]]


def _run_all_arms(sequences: List[DynamicSequence]) -> Dict[str, RunResult]:
    out: Dict[str, RunResult] = {}
    for name, factory in ARMS.items():
        per_seq: RunResult = {}
        for seq in sequences:
            controller = factory(seq)
            result, decisions = simulate_policy(seq, pricing_controller=controller)
            per_seq[seq.sequence_id] = (result, decisions)
        out[name] = per_seq
    return out


def _route_disagreement_rate(no_pricing: PolicyRunResult, arm: PolicyRunResult) -> float:
    total = len(no_pricing.choices)
    if total == 0:
        return 0.0
    disagree = sum(1 for a, b in zip(no_pricing.choices, arm.choices) if a != b)
    return disagree / total


def _lambda_engagement_rate(decisions: List[PolicyDecision], epsilon: float = 1e-6) -> float:
    if not decisions:
        return 0.0
    engaged = sum(1 for d in decisions if any(v > epsilon for v in d.lambda_price_before.values()))
    return engaged / len(decisions)


def _lambda_stats(decisions: List[PolicyDecision]) -> Dict[str, float]:
    values = [v for d in decisions for v in d.lambda_price_before.values()]
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _cell_axis_summary(
    sequences: List[DynamicSequence],
    runs: Dict[str, RunResult],
    hindsight: Dict[str, HindsightResult],
    params_by_cell: Dict[str, ScarcityParams],
) -> List[Dict[str, Any]]:
    by_cell: Dict[str, List[DynamicSequence]] = {}
    for seq in sequences:
        by_cell.setdefault(seq.scenario, []).append(seq)

    phase_map = []
    for cell_id, seqs in by_cell.items():
        seq_ids = [s.sequence_id for s in seqs]
        params = params_by_cell[cell_id]
        entry: Dict[str, Any] = {
            "cell_id": cell_id,
            "payoff_ratio": params.payoff_ratio,
            "budget_tightness": params.budget_tightness,
            "replenishment_mode": params.replenishment_mode,
            "timing": params.timing,
            "n_sequences": len(seqs),
            "arms": {},
        }
        no_pricing_results = [runs["no_pricing"][sid][0] for sid in seq_ids]
        for arm_name in ARMS:
            arm_results = [runs[arm_name][sid][0] for sid in seq_ids]
            metrics = regret_metrics(arm_results, hindsight)
            arm_entry: Dict[str, Any] = {
                "mean_regret": metrics["mean_regret"],
                "total_violation_count": metrics["total_violation_count"],
                "mean_terminal_unused_resources": metrics["mean_terminal_unused_resources"],
                "total_high_value_rejections": metrics["total_high_value_rejections"],
            }
            if arm_name != "no_pricing":
                deltas = paired_regret_deltas(arm_results, no_pricing_results, hindsight)
                disagreements = [
                    _route_disagreement_rate(runs["no_pricing"][sid][0], runs[arm_name][sid][0])
                    for sid in seq_ids
                ]
                engagements = [_lambda_engagement_rate(runs[arm_name][sid][1]) for sid in seq_ids]
                lambda_stats = [_lambda_stats(runs[arm_name][sid][1]) for sid in seq_ids]
                arm_entry["paired_delta_vs_no_pricing"] = statistics.mean(deltas)
                arm_entry["route_disagreement_rate"] = statistics.mean(disagreements)
                arm_entry["lambda_engagement_rate"] = statistics.mean(engagements)
                arm_entry["lambda_mean"] = statistics.mean(s["mean"] for s in lambda_stats)
                arm_entry["lambda_volatility"] = statistics.mean(s["std"] for s in lambda_stats)
                splits = [
                    conservation_depletion_split(
                        seq, runs[arm_name][seq.sequence_id][1], hindsight[seq.sequence_id]
                    )
                    for seq in seqs
                ]
                arm_entry["mean_regret_from_conservation"] = statistics.mean(
                    s["regret_from_conservation"] for s in splits
                )
                arm_entry["mean_regret_from_depletion"] = statistics.mean(
                    s["regret_from_depletion"] for s in splits
                )
            entry["arms"][arm_name] = arm_entry
        phase_map.append(entry)
    return phase_map


def _evaluate_h1_dormancy_under_slack(phase_map: List[Dict[str, Any]]) -> Dict[str, Any]:
    slack_cells = [c for c in phase_map if c["budget_tightness"] == 2.0]
    result: Dict[str, Any] = {"n_cells": len(slack_cells), "per_arm": {}}
    for arm_name in ARMS:
        if arm_name == "no_pricing":
            continue
        deltas = [c["arms"][arm_name]["paired_delta_vs_no_pricing"] for c in slack_cells]
        disagreements = [c["arms"][arm_name]["route_disagreement_rate"] for c in slack_cells]
        mean_abs_delta = statistics.mean(abs(d) for d in deltas)
        mean_disagreement = statistics.mean(disagreements)
        result["per_arm"][arm_name] = {
            "mean_abs_regret_delta": mean_abs_delta,
            "mean_route_disagreement_rate": mean_disagreement,
            "passed": bool(
                mean_abs_delta <= H1_REGRET_TOLERANCE
                and mean_disagreement <= H1_DISAGREEMENT_TOLERANCE
            ),
        }
    return result


def _evaluate_h2_benefit_under_scarcity(phase_map: List[Dict[str, Any]]) -> Dict[str, Any]:
    consequential = [
        c for c in phase_map if c["payoff_ratio"] > 1.0 and c["budget_tightness"] <= 1.1
    ]
    improving = [
        c
        for c in consequential
        if c["arms"][PRIMARY_ARM]["paired_delta_vs_no_pricing"] < MEANINGFUL_IMPROVEMENT_THRESHOLD
    ]
    distinct_configs = {
        (c["payoff_ratio"], c["replenishment_mode"], c["timing"]) for c in improving
    }
    return {
        "n_consequential_cells": len(consequential),
        "n_improving_cells": len(improving),
        "distinct_improving_configs": sorted(str(cfg) for cfg in distinct_configs),
        "n_distinct_improving_configs": len(distinct_configs),
        "passed": bool(len(distinct_configs) >= MIN_DISTINCT_CONFIGS_FOR_H2),
    }


def _evaluate_h3_interpretable_response(phase_map: List[Dict[str, Any]]) -> Dict[str, Any]:
    # For each fixed (replenishment_mode, timing) slice, sort by
    # (budget_tightness descending == less tight -> more tight, payoff_ratio
    # ascending) and record engagement/regret-delta trend; flag reversals
    # where tightening scarcity or raising payoff REDUCES engagement.
    by_slice: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for c in phase_map:
        key = (c["replenishment_mode"], c["timing"])
        by_slice.setdefault(key, []).append(c)

    reversals = []
    for key, cells in by_slice.items():
        ordered = sorted(cells, key=lambda c: (-c["budget_tightness"], c["payoff_ratio"]))
        engagements = [c["arms"][PRIMARY_ARM]["lambda_engagement_rate"] for c in ordered]
        for i in range(1, len(engagements)):
            if engagements[i] < engagements[i - 1] - 0.2:  # a real drop, not noise
                reversals.append(
                    {
                        "slice": str(key),
                        "from_cell": ordered[i - 1]["cell_id"],
                        "to_cell": ordered[i]["cell_id"],
                        "engagement_drop": engagements[i - 1] - engagements[i],
                    }
                )
    return {"n_slices": len(by_slice), "flagged_reversals": reversals}


def _evaluate_h4_boundary_behavior(phase_map: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_slice: Dict[Tuple[float, str, str], Dict[float, Dict[str, Any]]] = {}
    for c in phase_map:
        key = (c["payoff_ratio"], c["replenishment_mode"], c["timing"])
        by_slice.setdefault(key, {})[c["budget_tightness"]] = c

    jumps = []
    for key, by_tightness in by_slice.items():
        if 1.1 in by_tightness and 1.0 in by_tightness:
            regret_marginal = by_tightness[1.1]["arms"][PRIMARY_ARM]["mean_regret"]
            regret_severe = by_tightness[1.0]["arms"][PRIMARY_ARM]["mean_regret"]
            jumps.append(
                {
                    "config": str(key),
                    "regret_marginal": regret_marginal,
                    "regret_severe": regret_severe,
                    "jump": regret_severe - regret_marginal,
                }
            )
    jump_magnitudes = [abs(j["jump"]) for j in jumps]
    return {
        "boundary_jumps": jumps,
        "max_jump_magnitude": max(jump_magnitudes) if jump_magnitudes else 0.0,
    }


def _evaluate_h5_false_scarcity(
    phase_map: List[Dict[str, Any]], secondary_phase: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    false_scarcity_cells = [c for c in phase_map if c["payoff_ratio"] == 1.0]
    unused_deltas = []
    rejection_deltas = []
    for c in false_scarcity_cells:
        pacing_unused = c["arms"][PRIMARY_ARM]["mean_terminal_unused_resources"]
        no_pricing_unused = c["arms"]["no_pricing"]["mean_terminal_unused_resources"]
        unused_deltas.append(pacing_unused - no_pricing_unused)
        rejection_deltas.append(
            c["arms"][PRIMARY_ARM]["total_high_value_rejections"]
            - c["arms"]["no_pricing"]["total_high_value_rejections"]
        )
    mean_unused_delta = statistics.mean(unused_deltas) if unused_deltas else 0.0
    mean_rejection_delta = statistics.mean(rejection_deltas) if rejection_deltas else 0.0
    return {
        "n_false_scarcity_cells": len(false_scarcity_cells),
        "mean_unused_resource_delta_vs_no_pricing": mean_unused_delta,
        "mean_high_value_rejection_delta_vs_no_pricing": mean_rejection_delta,
        "passed": bool(abs(mean_unused_delta) <= H5_UNUSED_RESOURCE_TOLERANCE),
    }


def _evaluate_gate(
    phase_map: List[Dict[str, Any]],
    primary_sequences: List[DynamicSequence],
    primary_runs: Dict[str, RunResult],
    hindsight: Dict[str, HindsightResult],
    params_by_cell: Dict[str, ScarcityParams],
    h2: Dict[str, Any],
) -> Dict[str, Any]:
    def _is_consequential(seq: DynamicSequence) -> bool:
        params = params_by_cell[seq.scenario]
        return params.payoff_ratio > 1.0 and params.budget_tightness <= 1.1

    consequential_ids = [s.sequence_id for s in primary_sequences if _is_consequential(s)]
    pacing_results = [primary_runs[PRIMARY_ARM][sid][0] for sid in consequential_ids]
    no_pricing_results = [primary_runs["no_pricing"][sid][0] for sid in consequential_ids]
    deltas = paired_regret_deltas(pacing_results, no_pricing_results, hindsight)
    ci = bootstrap_ci(deltas)

    criterion_1 = ci["ci_high"] < 0.0
    criterion_2 = h2["passed"]

    total_violations_pacing = sum(
        primary_runs[PRIMARY_ARM][s.sequence_id][0].violation_count for s in primary_sequences
    )
    total_violations_no_pricing = sum(
        primary_runs["no_pricing"][s.sequence_id][0].violation_count for s in primary_sequences
    )
    criterion_3 = total_violations_pacing <= total_violations_no_pricing

    slack_and_false_scarcity_deltas = [
        c["arms"][PRIMARY_ARM]["paired_delta_vs_no_pricing"]
        for c in phase_map
        if c["budget_tightness"] == 2.0 or c["payoff_ratio"] == 1.0
    ]
    criterion_4 = (
        statistics.mean(abs(d) for d in slack_and_false_scarcity_deltas)
        <= GATE4_NON_INFERIORITY_TOLERANCE
    )

    n_improving_sequences = h2["n_improving_cells"] * 3  # 3 seeds per cell
    criterion_5 = n_improving_sequences >= MIN_SEQUENCES_FOR_GATE5

    h4 = _evaluate_h4_boundary_behavior(phase_map)
    criterion_6 = h4["max_jump_magnitude"] < MAX_BOUNDARY_JUMP

    mixture_weights = {c["cell_id"]: 1.0 / len(phase_map) for c in phase_map}
    pacing_mixture_regret = sum(
        mixture_weights[c["cell_id"]] * c["arms"][PRIMARY_ARM]["mean_regret"] for c in phase_map
    )
    no_pricing_mixture_regret = sum(
        mixture_weights[c["cell_id"]] * c["arms"]["no_pricing"]["mean_regret"] for c in phase_map
    )
    criterion_7 = pacing_mixture_regret < no_pricing_mixture_regret

    criteria = {
        "1_significant_in_consequential_region": criterion_1,
        "2_holds_across_multiple_configs": criterion_2,
        "3_no_additional_violations": criterion_3,
        "4_non_inferior_in_slack_and_false_scarcity": criterion_4,
        "5_not_driven_by_isolated_sequences": criterion_5,
        "6_stable_near_boundary": criterion_6,
        "7_lower_aggregate_mixture_regret": criterion_7,
    }
    return {
        "consequential_region_paired_delta_ci": ci,
        "criteria": criteria,
        "n_criteria_passed": sum(criteria.values()),
        "passed": bool(all(criteria.values())),
        "pacing_mixture_regret": pacing_mixture_regret,
        "no_pricing_mixture_regret": no_pricing_mixture_regret,
    }


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    primary_sequences = generate_primary_dataset(seed=SEED)
    secondary_datasets = generate_secondary_dataset(seed=SEED)
    params_by_cell = {p.cell_id(): p for p in primary_grid()}

    print(f"{len(primary_sequences)} primary sequences across {len(params_by_cell)} cells")

    hindsight: Dict[str, HindsightResult] = {
        seq.sequence_id: compute_hindsight_optimum(seq) for seq in primary_sequences
    }
    for seqs in secondary_datasets.values():
        for seq in seqs:
            hindsight[seq.sequence_id] = compute_hindsight_optimum(seq)

    primary_runs = _run_all_arms(primary_sequences)
    secondary_runs = {axis: _run_all_arms(seqs) for axis, seqs in secondary_datasets.items()}

    phase_map = _cell_axis_summary(primary_sequences, primary_runs, hindsight, params_by_cell)

    h1 = _evaluate_h1_dormancy_under_slack(phase_map)
    h2 = _evaluate_h2_benefit_under_scarcity(phase_map)
    h3 = _evaluate_h3_interpretable_response(phase_map)
    h4 = _evaluate_h4_boundary_behavior(phase_map)
    h5 = _evaluate_h5_false_scarcity(phase_map, {})
    gate = _evaluate_gate(phase_map, primary_sequences, primary_runs, hindsight, params_by_cell, h2)

    secondary_summary: Dict[str, Any] = {}
    for axis, seqs in secondary_datasets.items():
        runs = secondary_runs[axis]
        no_pricing_results = [runs["no_pricing"][s.sequence_id][0] for s in seqs]
        pacing_results = [runs[PRIMARY_ARM][s.sequence_id][0] for s in seqs]
        deltas = paired_regret_deltas(pacing_results, no_pricing_results, hindsight)
        unused = [
            runs[PRIMARY_ARM][s.sequence_id][0].terminal_remaining.get("budget", 0.0) for s in seqs
        ]
        secondary_summary[axis] = {
            "n_sequences": len(seqs),
            "mean_paired_delta_vs_no_pricing": statistics.mean(deltas),
            "mean_terminal_unused_budget": statistics.mean(unused),
        }

    report: Dict[str, Any] = {
        "schema": "compitum.scarcity-phase-study-report/v1",
        "design": (
            "observation-only; frozen tranche-4 controller parameters, no "
            "retuning; primary factorial grid (payoff_ratio x budget_tightness "
            "x replenishment_mode x timing, 72 cells x 3 seeds); secondary "
            "one-at-a-time sweeps against a fixed reference cell"
        ),
        "primary_arm": PRIMARY_ARM,
        "frozen_parameters": {
            "reactive": {"eta": DUAL_ETA, "lambda_max": DUAL_LAMBDA_MAX},
            "pacing": FROZEN_PACING,
            "pacing_hysteresis": FROZEN_HYSTERESIS,
            "pacing_asymmetric": FROZEN_ASYMMETRIC,
            "pacing_bounded_smoothed": FROZEN_BOUNDED_SMOOTHED,
        },
        "n_primary_cells": len(params_by_cell),
        "n_primary_sequences": len(primary_sequences),
        "phase_map": phase_map,
        "hypotheses": {
            "H1_dormancy_under_slack": h1,
            "H2_benefit_under_scarcity": h2,
            "H3_interpretable_response": h3,
            "H4_boundary_behavior": h4,
            "H5_robustness_to_false_scarcity": h5,
        },
        "secondary_sweeps": secondary_summary,
        "activation_gate": gate,
        "total_elapsed_seconds": time.perf_counter() - started,
    }

    out_path = ARTIFACTS / "phase_study_report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    out_path.write_text(rendered, encoding="utf-8", newline="")
    print(
        json.dumps(
            {
                "hypotheses_summary": {
                    k: v.get("passed")
                    for k, v in report["hypotheses"].items()
                    if isinstance(v, dict) and "passed" in v
                },
                "gate": gate["criteria"],
                "gate_passed": gate["passed"],
            },
            indent=2,
        )
    )
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
