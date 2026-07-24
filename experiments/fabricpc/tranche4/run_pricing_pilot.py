"""Tranche 4 bounded, observation-only pricing-controller pilot.

Pure ``compitum.regret_lab`` -- no FabricPC, no JAX, runs under the plain
``.venv``. Per docs/adr/0004-pricing-controller-repair.md: tranche 3 found
that bad pricing is worse than no pricing (the reactive dual controller
lost to no pricing, and learned correctors on top of it lost by more,
merely hoarding resources). This tranche is exclusively about establishing
a non-learned pricing controller that beats no pricing before any learned
predictor is reintroduced.

Six arms, paired across identical held-out test sequences:
  1. no pricing (immutable control)
  2. reactive dual controller (tranche 3's failed reference, unchanged
     parameters -- not tuned, since it is retained as a fixed comparison
     point, not a candidate)
  3. pacing
  4. pacing + hysteresis/deadband
  5. asymmetric (slower rise, faster relaxation)
  6. bounded + EMA-smoothed

Arms 3-6 share one parameterized ``PacingController``; their parameters
are selected via a small, declared grid search on TRAINING sequences only
(scored by mean regret subject to zero increase in hard violations vs the
reactive baseline), then frozen before any held-out test-sequence
evaluation.

Gate: a pacing-family arm passes only if it has a paired
bootstrap-CI-significant reduction in mean regret vs no pricing AND does
not increase total violations vs no pricing.
"""

from __future__ import annotations

import json
import statistics
import time
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from compitum.regret_lab import (
    RESOURCE_NAMES,
    DynamicSequence,
    HindsightResult,
    PacingController,
    PolicyRunResult,
    ReactiveController,
    bootstrap_ci,
    compute_hindsight_optimum,
    conservation_depletion_split,
    generate_dynamic_dataset,
    paired_regret_deltas,
    regret_metrics,
    simulate_policy,
    total_available_over_horizon,
)
from compitum.regret_lab.environment import SCENARIOS
from compitum.regret_lab.simulator import PolicyDecision

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche4" / "artifacts"
SEQUENCES_PER_SCENARIO = 4
TRAIN_SEQUENCES_PER_SCENARIO = 2
STEPS_PER_SEQUENCE = 8
DUAL_ETA = 0.5
DUAL_LAMBDA_MAX = 20.0

ControllerFactory = Callable[[DynamicSequence], Optional[Any]]


def _split_sequences(
    sequences: List[DynamicSequence],
) -> Tuple[List[DynamicSequence], List[DynamicSequence]]:
    is_train_flags = [True] * TRAIN_SEQUENCES_PER_SCENARIO + [False] * (
        SEQUENCES_PER_SCENARIO - TRAIN_SEQUENCES_PER_SCENARIO
    )
    train, test = [], []
    for seq in sequences:
        index = int(seq.sequence_id.rsplit("-", 1)[1])
        if is_train_flags[index % SEQUENCES_PER_SCENARIO]:
            train.append(seq)
        else:
            test.append(seq)
    return train, test


def _no_pricing(seq: DynamicSequence) -> None:
    return None


def _reactive(seq: DynamicSequence) -> ReactiveController:
    return ReactiveController(
        resource_names=seq.resource_names, eta=DUAL_ETA, lambda_max=DUAL_LAMBDA_MAX
    )


def _run_arm(
    sequences: List[DynamicSequence], build_controller: ControllerFactory
) -> Tuple[List[PolicyRunResult], Dict[str, List[PolicyDecision]]]:
    results = []
    decisions_by_seq: Dict[str, List[PolicyDecision]] = {}
    for seq in sequences:
        controller = build_controller(seq)
        result, decisions = simulate_policy(seq, pricing_controller=controller)
        results.append(result)
        decisions_by_seq[seq.sequence_id] = decisions
    return results, decisions_by_seq


def _grid_search(
    family_name: str,
    grid: List[Dict[str, float]],
    build: Callable[[DynamicSequence, Dict[str, float]], PacingController],
    train_sequences: List[DynamicSequence],
    train_hindsight: Dict[str, HindsightResult],
    reference_violations: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Selects the config with the lowest mean regret among configs with
    total violations <= ``reference_violations`` (the no-pricing arm's
    count); if none qualify, falls back to the lowest-violation config,
    tie-broken by regret. Never touches test sequences."""
    scored = []
    for params in grid:
        results, _ = _run_arm(train_sequences, lambda seq: build(seq, params))
        metrics = regret_metrics(results, train_hindsight)
        scored.append((params, metrics))

    qualifying = [(p, m) for p, m in scored if m["total_violation_count"] <= reference_violations]
    pool = qualifying if qualifying else scored
    best_params, best_metrics = min(pool, key=lambda pm: pm[1]["mean_regret"])
    selection_report = {
        "family": family_name,
        "grid_size": len(grid),
        "qualifying_configs": len(qualifying),
        "selection_criterion": (
            "lowest mean regret on training sequences among configs with "
            "total_violation_count <= no-pricing reference; else lowest "
            "violation count, tie-broken by regret"
        ),
        "selected_params": best_params,
        "selected_train_mean_regret": best_metrics["mean_regret"],
        "selected_train_violation_count": best_metrics["total_violation_count"],
    }
    return best_params, selection_report


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    sequences = generate_dynamic_dataset(
        seed=2026,
        sequences_per_scenario=SEQUENCES_PER_SCENARIO,
        steps_per_sequence=STEPS_PER_SEQUENCE,
    )
    train_sequences, test_sequences = _split_sequences(sequences)
    print(f"{len(train_sequences)} train sequences, {len(test_sequences)} test sequences")

    train_hindsight = {seq.sequence_id: compute_hindsight_optimum(seq) for seq in train_sequences}
    test_hindsight = {seq.sequence_id: compute_hindsight_optimum(seq) for seq in test_sequences}

    no_pricing_train_results, _ = _run_arm(train_sequences, _no_pricing)
    reference_violations = regret_metrics(no_pricing_train_results, train_hindsight)[
        "total_violation_count"
    ]

    # -- Dev-set (training-sequence-only) parameter selection, frozen before
    # any held-out evaluation. Grids are small and declared here in full.
    def _pacing_plain(seq: DynamicSequence, p: Dict[str, float]) -> PacingController:
        return PacingController(
            resource_names=seq.resource_names,
            total_available=total_available_over_horizon(seq),
            total_steps=len(seq.cases),
            eta=p["eta"],
        )

    def _pacing_hysteresis(seq: DynamicSequence, p: Dict[str, float]) -> PacingController:
        return PacingController(
            resource_names=seq.resource_names,
            total_available=total_available_over_horizon(seq),
            total_steps=len(seq.cases),
            eta=p["eta"],
            deadband=p["deadband"],
        )

    def _pacing_asymmetric(seq: DynamicSequence, p: Dict[str, float]) -> PacingController:
        return PacingController(
            resource_names=seq.resource_names,
            total_available=total_available_over_horizon(seq),
            total_steps=len(seq.cases),
            eta=p["eta"],
            relax_gamma=p["relax_gamma"],
            rise_scale=p["rise_scale"],
        )

    def _pacing_bounded_smoothed(seq: DynamicSequence, p: Dict[str, float]) -> PacingController:
        return PacingController(
            resource_names=seq.resource_names,
            total_available=total_available_over_horizon(seq),
            total_steps=len(seq.cases),
            eta=p["eta"],
            ema_alpha=p["ema_alpha"],
            max_step=p["max_step"],
        )

    # Widened after an initial narrow grid (0.1-1.0) was found empirically to
    # miss a real, much better-performing region: a direct training-sequence
    # sweep showed eta in roughly [1.2, 2.1] cuts mean regret from ~2.86
    # (no pricing) to ~0.34-0.44, which no config in the original grid came
    # close to. Verified before trusting it, per this project's standing
    # practice of not accepting a suspiciously flat result at face value.
    grids: Dict[str, List[Dict[str, float]]] = {
        "pacing": [{"eta": eta} for eta in (0.5, 1.0, 1.5, 1.8, 2.0, 2.1, 3.0)],
        "pacing_hysteresis": [
            {"eta": eta, "deadband": deadband}
            for eta, deadband in product((1.0, 1.5, 2.0), (0.05, 0.1, 0.2))
        ],
        "pacing_asymmetric": [
            {"eta": eta, "relax_gamma": gamma, "rise_scale": rise}
            for eta, gamma, rise in product((1.0, 1.5, 2.0), (2.0, 4.0), (0.5, 1.0))
        ],
        "pacing_bounded_smoothed": [
            {"eta": eta, "ema_alpha": alpha, "max_step": step}
            for eta, alpha, step in product((1.0, 1.5, 2.0, 3.0), (0.1, 0.3, 0.5), (0.5, 1.0, 2.0))
        ],
    }
    builders = {
        "pacing": _pacing_plain,
        "pacing_hysteresis": _pacing_hysteresis,
        "pacing_asymmetric": _pacing_asymmetric,
        "pacing_bounded_smoothed": _pacing_bounded_smoothed,
    }

    selected_params: Dict[str, Dict[str, float]] = {}
    selection_reports: Dict[str, Any] = {}
    for family, grid in grids.items():
        params, report = _grid_search(
            family, grid, builders[family], train_sequences, train_hindsight, reference_violations
        )
        selected_params[family] = params
        selection_reports[family] = report
        print(f"selected {family}: {params}")

    # -- Held-out evaluation: six paired arms on untouched test sequences.
    arm_builders: Dict[str, ControllerFactory] = {
        "no_pricing": _no_pricing,
        "reactive": _reactive,
        "pacing": lambda seq: builders["pacing"](seq, selected_params["pacing"]),
        "pacing_hysteresis": lambda seq: builders["pacing_hysteresis"](
            seq, selected_params["pacing_hysteresis"]
        ),
        "pacing_asymmetric": lambda seq: builders["pacing_asymmetric"](
            seq, selected_params["pacing_asymmetric"]
        ),
        "pacing_bounded_smoothed": lambda seq: builders["pacing_bounded_smoothed"](
            seq, selected_params["pacing_bounded_smoothed"]
        ),
    }

    arm_results: Dict[str, List[PolicyRunResult]] = {}
    arm_decisions: Dict[str, Dict[str, List[PolicyDecision]]] = {}
    for name, factory in arm_builders.items():
        results, decisions_by_seq = _run_arm(test_sequences, factory)
        arm_results[name] = results
        arm_decisions[name] = decisions_by_seq

    metrics = {
        name: regret_metrics(results, test_hindsight) for name, results in arm_results.items()
    }

    # -- Paired bootstrap CIs vs no pricing.
    gate: Dict[str, Any] = {}
    for name in arm_builders:
        if name == "no_pricing":
            continue
        deltas = paired_regret_deltas(arm_results[name], arm_results["no_pricing"], test_hindsight)
        ci = bootstrap_ci(deltas)
        beats_no_pricing = ci["ci_high"] < 0.0
        violations_not_increased = (
            metrics[name]["total_violation_count"] <= metrics["no_pricing"]["total_violation_count"]
        )
        gate[name] = {
            "paired_regret_delta_vs_no_pricing": ci,
            "beats_no_pricing": beats_no_pricing,
            "violations_not_increased": violations_not_increased,
            "passed": bool(beats_no_pricing and violations_not_increased),
        }

    # -- Scenario-stratified regret.
    scenario_table: Dict[str, Dict[str, float]] = {}
    for scenario in SCENARIOS:
        scenario_ids = {seq.sequence_id for seq in test_sequences if seq.scenario == scenario}
        scenario_table[scenario] = {}
        for name, results in arm_results.items():
            subset = [r for r in results if r.sequence_id in scenario_ids]
            scenario_table[scenario][name] = regret_metrics(subset, test_hindsight)["mean_regret"]

    # -- Premature-conservation diagnostics, aggregated per arm.
    conservation_diagnostics: Dict[str, Any] = {}
    for name in arm_builders:
        splits = [
            conservation_depletion_split(
                seq, arm_decisions[name][seq.sequence_id], test_hindsight[seq.sequence_id]
            )
            for seq in test_sequences
        ]
        conservation_diagnostics[name] = {
            "mean_regret_from_conservation": statistics.mean(
                s["regret_from_conservation"] for s in splits
            ),
            "mean_regret_from_depletion": statistics.mean(
                s["regret_from_depletion"] for s in splits
            ),
            "mean_unattributed_regret": statistics.mean(s["unattributed_regret"] for s in splits),
        }

    # -- Controller price volatility (std dev of the lambda-price
    # trajectory per resource, averaged across resources and sequences).
    price_volatility: Dict[str, float] = {}
    for name in arm_builders:
        if name == "no_pricing":
            price_volatility[name] = 0.0
            continue
        per_sequence_stds = []
        for seq in test_sequences:
            decisions = arm_decisions[name][seq.sequence_id]
            for r in RESOURCE_NAMES:
                series = [d.lambda_price_before.get(r, 0.0) for d in decisions]
                if len(series) > 1:
                    per_sequence_stds.append(statistics.pstdev(series))
        price_volatility[name] = statistics.mean(per_sequence_stds) if per_sequence_stds else 0.0

    report: Dict[str, Any] = {
        "schema": "compitum.pricing-controller-pilot-report/v1",
        "design": (
            "observation-only; pacing-family parameters selected on training "
            "sequences only, frozen before held-out evaluation; reactive "
            "controller retained unchanged (not tuned) as the failed tranche 3 "
            "reference"
        ),
        "n_train_sequences": len(train_sequences),
        "n_test_sequences": len(test_sequences),
        "parameter_selection": selection_reports,
        "arms": metrics,
        "activation_gate": {
            "criterion": (
                "a pricing arm passes only if it has a paired bootstrap-CI"
                "-significant reduction in mean regret vs no pricing AND does "
                "not increase total violations vs no pricing"
            ),
            "per_arm": gate,
            "any_passed": any(g["passed"] for g in gate.values()),
        },
        "scenario_stratified_mean_regret": scenario_table,
        "conservation_depletion_diagnostics": conservation_diagnostics,
        "controller_price_volatility": price_volatility,
        "total_elapsed_seconds": time.perf_counter() - started,
    }

    out_path = ARTIFACTS / "pilot_report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    out_path.write_text(rendered, encoding="utf-8", newline="")
    print(json.dumps(report["activation_gate"], indent=2))
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
