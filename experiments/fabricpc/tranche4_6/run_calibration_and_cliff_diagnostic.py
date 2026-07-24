"""Tranche 4.6: corrected-slack rerun + extreme-payoff cliff diagnostic.

Pure ``compitum.regret_lab`` -- no FabricPC, no JAX, no controller tuning.
Per the user's tranche 4.6 brief: resolve two open questions from tranche
4.5 before any FabricPC work.

Part 1 (recalibration): tranche 4.5's H1 failure (non-dormancy in
near/mid-timing slack cells) was traced to `budget_tightness` being
calibrated against a fully-conservative reference rate rather than the
environment's natural spend-preferring behavior. This reruns exactly the
affected cells with the corrected calibration
(`generate_corrected_slack_dataset`), same frozen pacing parameters, and
reports whether disagreement/regret-impact disappears.

Part 2 (cliff diagnostic): tranche 4.5's H4 found a full-magnitude regret
jump between budget_tightness=1.1 and 1.0 at payoff_ratio=10.0 in 2 of 6
configurations. This densely samples ABSOLUTE initial budget (not the
budget_tightness ratio, which aliases onto the same quantized value at
fine resolution -- see the "sampling resolution" note in the report) at
the finest available grid resolution (GRID_UNIT=0.25) around the boundary,
for both the flagged and unflagged configs, recording regret, choices,
lambda trajectory, and remaining-budget trajectory at each point.

Finding, verified directly before use here: for `opportunity_prevalence=
"rare"` (every primary-grid cell), the RNG is never consumed inside
`build_scarcity_sequence`, so tranche 4.5's "3 seeds per cell" are
byte-identical duplicates, not independent draws -- this diagnostic uses a
single representative sequence per configuration rather than wastefully
repeating identical runs, and this finding is reported honestly rather
than silently carried forward.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from compitum.regret_lab import (
    GRID_UNIT,
    PacingController,
    ReactiveController,
    compute_hindsight_optimum,
    generate_corrected_slack_dataset,
    generate_primary_dataset,
    paired_regret_deltas,
    regret_metrics,
    simulate_policy,
    total_available_over_horizon,
)
from compitum.regret_lab.scarcity_scenarios import (
    CONSERVE_RATE,
    OPPORTUNITY_COST,
    ScarcityParams,
    _timing_step,
    build_scarcity_sequence,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche4_6" / "artifacts"
SEED = 4242
STEPS = 12

DUAL_ETA = 0.5
DUAL_LAMBDA_MAX = 20.0
FROZEN_PACING = {"eta": 1.8}


def _no_pricing(seq):
    return None


def _reactive(seq):
    return ReactiveController(
        resource_names=seq.resource_names, eta=DUAL_ETA, lambda_max=DUAL_LAMBDA_MAX
    )


def _pacing(seq):
    return PacingController(
        resource_names=seq.resource_names,
        total_available=total_available_over_horizon(seq),
        total_steps=len(seq.cases),
        **FROZEN_PACING,
    )


def _part1_corrected_slack_rerun() -> Dict[str, Any]:
    original_all = generate_primary_dataset(seed=SEED)
    original_near_mid = {
        s.sequence_id: s
        for s in original_all
        for _ in [None]
        if _cell_timing(s.scenario) in ("near", "mid")
    }
    corrected = generate_corrected_slack_dataset(seed=SEED)

    def _run(sequences):
        hindsight = {s.sequence_id: compute_hindsight_optimum(s) for s in sequences}
        no_pricing = {}
        pacing = {}
        for s in sequences:
            r0, _ = simulate_policy(s)
            r1, d1 = simulate_policy(s, pricing_controller=_pacing(s))
            no_pricing[s.sequence_id] = r0
            pacing[s.sequence_id] = (r1, d1)
        return hindsight, no_pricing, pacing

    orig_seqs = list(original_near_mid.values())
    orig_hindsight, orig_no_pricing, orig_pacing = _run(orig_seqs)
    corr_hindsight, corr_no_pricing, corr_pacing = _run(corrected)

    def _summarize(sequences, hindsight, no_pricing, pacing, label):
        slack_ids = [s.sequence_id for s in sequences if s.budget_tightness_label == "slack"]
        all_ids = [s.sequence_id for s in sequences]

        def _stats(ids):
            no_pricing_results = [no_pricing[i] for i in ids]
            pacing_results = [pacing[i][0] for i in ids]
            metrics_pacing = regret_metrics(pacing_results, hindsight)
            deltas = paired_regret_deltas(pacing_results, no_pricing_results, hindsight)
            disagreements = []
            for i in ids:
                np_choices = no_pricing[i].choices
                p_choices = pacing[i][0].choices
                total = len(np_choices)
                disagree = sum(1 for a, b in zip(np_choices, p_choices) if a != b)
                disagreements.append(disagree / total if total else 0.0)
            return {
                "n_sequences": len(ids),
                "mean_abs_regret_delta": float(np.mean([abs(d) for d in deltas]))
                if deltas
                else 0.0,
                "mean_route_disagreement_rate": float(np.mean(disagreements))
                if disagreements
                else 0.0,
                "mean_regret_pacing": metrics_pacing["mean_regret"],
            }

        return {
            "label": label,
            "slack_cells": _stats(slack_ids),
            "all_near_mid_cells": _stats(all_ids),
        }

    return {
        "original": _summarize(
            [_Tagged(s) for s in orig_seqs],
            orig_hindsight,
            orig_no_pricing,
            orig_pacing,
            "original",
        ),
        "corrected": _summarize(
            [_Tagged(s) for s in corrected],
            corr_hindsight,
            corr_no_pricing,
            corr_pacing,
            "corrected",
        ),
    }


def _cell_timing(cell_id: str) -> str:
    for fragment in cell_id.split("-"):
        if fragment.startswith("t") and fragment[1:] in ("near", "mid", "final"):
            return fragment[1:]
    return ""


class _Tagged:
    """Wraps a DynamicSequence with its budget_tightness axis label
    ('slack' | 'marginal' | 'severe') parsed from its cell id, for the
    part-1 summary above without touching DynamicSequence itself."""

    def __init__(self, seq):
        self.sequence_id = seq.sequence_id
        tag = "unknown"
        if "bt2.0" in seq.scenario:
            tag = "slack"
        elif "bt1.1" in seq.scenario:
            tag = "marginal"
        elif "bt1.0" in seq.scenario:
            tag = "severe"
        self.budget_tightness_label = tag


def _part2_cliff_diagnostic() -> Dict[str, Any]:
    flagged = [("none", "mid"), ("partial", "near")]
    unflagged = [("none", "near"), ("none", "final"), ("partial", "mid"), ("partial", "final")]

    results = []
    for replenishment_mode, timing in flagged + unflagged:
        t_opp = _timing_step(timing, STEPS)
        min_budget_needed = t_opp * CONSERVE_RATE + OPPORTUNITY_COST
        # Densely sample absolute budget at the finest available grid
        # resolution (GRID_UNIT) from just below the marginal (1.1x) point
        # down through and below the severe (1.0x) point.
        low = min_budget_needed * 0.95
        high = min_budget_needed * 1.15
        n_points = int(round((high - low) / GRID_UNIT)) + 1
        budgets = [low + i * GRID_UNIT for i in range(n_points)]

        points = []
        for absolute_budget in budgets:
            params = ScarcityParams(
                payoff_ratio=10.0,
                budget_tightness=absolute_budget / min_budget_needed,
                replenishment_mode=replenishment_mode,
                timing=timing,
            )
            rng = np.random.default_rng(0)  # unused for opportunity_prevalence="rare"
            seq = build_scarcity_sequence(
                rng, params, STEPS, f"cliff-{replenishment_mode}-{timing}"
            )
            hindsight = compute_hindsight_optimum(seq)
            no_pricing_result, no_pricing_decisions = simulate_policy(seq)
            pacing_result, pacing_decisions = simulate_policy(seq, pricing_controller=_pacing(seq))
            points.append(
                {
                    "initial_budget": seq.initial_budget["budget"],
                    "budget_tightness_ratio": absolute_budget / min_budget_needed,
                    "hindsight_value": hindsight.value,
                    "no_pricing": {
                        "regret": hindsight.value - no_pricing_result.cumulative_utility,
                        "choices": no_pricing_result.choices,
                        "opportunity_captured": "opportunity" in no_pricing_result.choices,
                    },
                    "pacing": {
                        "regret": hindsight.value - pacing_result.cumulative_utility,
                        "choices": pacing_result.choices,
                        "opportunity_captured": "opportunity" in pacing_result.choices,
                        "lambda_trajectory": [
                            d.lambda_price_before.get("budget", 0.0) for d in pacing_decisions
                        ],
                        "remaining_trajectory": [
                            d.remaining_before.get("budget", 0.0) for d in pacing_decisions
                        ],
                    },
                }
            )

        # Identify the transition band for each policy: the largest single
        # -step budget increment across which opportunity_captured flips.
        def _transition_band(points, key):
            captured = [p[key]["opportunity_captured"] for p in points]
            for i in range(1, len(captured)):
                if captured[i] != captured[i - 1]:
                    return {
                        "from_budget": points[i - 1]["initial_budget"],
                        "to_budget": points[i]["initial_budget"],
                        "from_captured": captured[i - 1],
                        "to_captured": captured[i],
                    }
            return None

        results.append(
            {
                "config": {"replenishment_mode": replenishment_mode, "timing": timing},
                "flagged_in_tranche_4_5": (replenishment_mode, timing) in flagged,
                "min_budget_needed": min_budget_needed,
                "n_points_sampled": len(points),
                "no_pricing_transition": _transition_band(points, "no_pricing"),
                "pacing_transition": _transition_band(points, "pacing"),
                "points": points,
            }
        )
    return {"grid_unit": GRID_UNIT, "configs": results}


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("part 1: corrected-slack rerun...")
    part1 = _part1_corrected_slack_rerun()
    print(json.dumps(part1, indent=2, default=str)[:2000])

    print("part 2: extreme-payoff cliff diagnostic...")
    part2 = _part2_cliff_diagnostic()

    report = {
        "schema": "compitum.tranche4-6-diagnostic-report/v1",
        "part1_corrected_slack_rerun": part1,
        "part2_cliff_diagnostic": part2,
        "methodology_note": (
            "opportunity_prevalence='rare' (every primary-grid cell) never "
            "consumes its rng argument in build_scarcity_sequence, so "
            "tranche 4.5's '3 seeds per cell' are byte-identical duplicates "
            "for the primary grid, not independent draws. Verified directly "
            "before writing this diagnostic. Does not invalidate tranche "
            "4.5's cross-CONFIGURATION findings (which vary declared axes, "
            "not RNG), but the seed count should not be read as adding "
            "independent statistical power for rare-prevalence cells."
        ),
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    out_path = ARTIFACTS / "calibration_and_cliff_report.json"
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline=""
    )
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
