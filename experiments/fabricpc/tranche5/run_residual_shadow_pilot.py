"""Tranche 5: seven-arm paired offline FabricPC residual shadow pilot.

Runs under ``.venv-fabricpc``. Per docs/adr/0006-fabricpc-residual-shadow-pricing.md:
FabricPC is reintroduced strictly as a bounded, gated corrector of the
frozen tranche-4/4.6 pacing controller's price, never as its replacement.
The frozen pacing controller (arm 2) -- not the reactive controller -- is
the baseline every learned arm must beat.

Dataset: the primary scarcity grid (72 cells), ONE sequence per cell
(``seeds_per_cell=1``): tranche 4.6 found ``opportunity_prevalence="rare"``
never consumes its RNG, so extra "seeds" would be duplicates, not
independent draws -- reported honestly there, applied honestly here by
simply not manufacturing false replication. Cells are split by index
(``i % 3 == 0`` -> test, 24 cells; else train, 48 cells) so train/test
separation is by genuinely different environment configuration, not by
RNG seed.

Training (train cells only, never touched again after this): a recorder
``ResidualPricingController`` with a permanently-zero predictor runs
alongside frozen pacing (so recorded decisions exactly match plain frozen
pacing -- a shadow data-collection pass, never influencing anything),
capturing each step's declared window and the frozen controller's own
``lambda_base``. The oracle-compatible interval (``residual_target.py``)
against the hindsight oracle's own per-step choice gives the target;
infeasible rows are excluded, never treated as zero. Three predictors are
then fit offline and frozen: a plain ridge regression directly on the
flattened window (arm 3), and two ridge regressions on FabricPC trajectory
-summary features (terminal-only for arm 4, full window for arm 5) of
that same window observed through the JAX-side windowed observer.

Seven held-out test arms, paired: (1) no pricing; (2) frozen pacing;
(3) frozen pacing + non-FabricPC windowed ridge; (4) frozen pacing +
FabricPC terminal-state residual; (5) frozen pacing + FabricPC trajectory
residual; (6) frozen pacing + shuffled-trajectory FabricPC residual
(negative control); (7) frozen pacing + FabricPC trajectory residual with
the regional-scope gate forcibly held open everywhere.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fabricpc_residual_observer import DEFAULT_CHECKOUT, MAX_WINDOW, observe_window  # noqa: E402

from compitum.constraint_oracle.experiment import (  # noqa: E402
    shuffle_raw_steps,
    terminal_features_from_evidence,
    trajectory_features_from_evidence,
)
from compitum.regret_lab import (  # noqa: E402
    RESIDUAL_CHANNEL_DIMENSION,
    DynamicSequence,
    HindsightResult,
    PacingController,
    PolicyRunResult,
    ResidualPricingController,
    RidgeModel,
    bootstrap_ci,
    compute_hindsight_optimum,
    compute_oracle_compatible_interval,
    conservation_depletion_split,
    fit_ridge,
    flatten_window,
    generate_primary_dataset,
    oracle_price_residual,
    paired_regret_deltas,
    predict_ridge,
    regret_metrics,
    simulate_policy,
    total_available_over_horizon,
)
from compitum.regret_lab.residual_pricing import ResidualCorrectionRecord  # noqa: E402
from compitum.regret_lab.simulator import PolicyDecision  # noqa: E402
from compitum.trajectory.evidence import build_evidence  # noqa: E402
from compitum.trajectory.types import ObservationStatus, TrajectoryRequest  # noqa: E402

ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche5" / "artifacts"
SEED = 4242
FROZEN_PACING_ETA = 1.8  # tranche 4's selected plain-pacing configuration, unchanged
MAX_CORRECTION_MAGNITUDE = 2.0
SCARCITY_GATE_THRESHOLD = 0.01
TEST_CELL_MODULUS = 3  # cell index % 3 == 0 -> test, else train


def _pacing(seq: DynamicSequence) -> PacingController:
    return PacingController(
        resource_names=seq.resource_names,
        total_available=total_available_over_horizon(seq),
        total_steps=len(seq.cases),
        eta=FROZEN_PACING_ETA,
    )


def _scarcity_gate(context: Any, lambda_base: float) -> bool:
    return lambda_base > SCARCITY_GATE_THRESHOLD


def _gate_always_open(context: Any, lambda_base: float) -> bool:
    return True


def _split_sequences(
    sequences: List[DynamicSequence],
) -> Tuple[List[DynamicSequence], List[DynamicSequence]]:
    train = [s for i, s in enumerate(sequences) if i % TEST_CELL_MODULUS != 0]
    test = [s for i, s in enumerate(sequences) if i % TEST_CELL_MODULUS == 0]
    return train, test


def _collect_training_rows(
    train_sequences: List[DynamicSequence],
) -> Tuple[List[np.ndarray], List[float], List[str]]:
    """Runs the zero-correction recorder alongside frozen pacing on every
    training sequence; returns (window, target, row_id) for every row with
    a feasible oracle-compatible interval."""
    windows: List[np.ndarray] = []
    targets: List[float] = []
    row_ids: List[str] = []
    for seq in train_sequences:
        recorder = ResidualPricingController(
            base=_pacing(seq), predict_residual=lambda window: 0.0, max_correction_magnitude=0.0
        )
        simulate_policy(seq, pricing_controller=recorder)
        hindsight = compute_hindsight_optimum(seq)
        # pricing_controller.update() (and therefore a record) is only
        # produced on steps where a model was actually chosen -- a "defer"
        # step never calls update(), so records must be looked up by their
        # own .step field, never assumed to align with list index.
        records_by_step = {r.step: r for r in recorder.records}
        for t, case in enumerate(seq.cases):
            record = records_by_step.get(t)
            if record is None:
                continue
            interval = compute_oracle_compatible_interval(case, hindsight.choices[t])
            target = oracle_price_residual(interval, record.lambda_base)
            if target is None:
                continue
            windows.append(record.window_snapshot)
            targets.append(target)
            row_ids.append(f"train-{seq.sequence_id}-step{t}")
    return windows, targets, row_ids


def _observe_evidence(
    row_id: str,
    window: List[np.ndarray],
    terminal_only: bool,
    failures: List[Dict[str, Any]],
    latencies: List[float],
    shuffle_seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    source_window = [window[-1]] if terminal_only else window
    flattened = flatten_window(source_window, MAX_WINDOW, RESIDUAL_CHANNEL_DIMENSION)
    try:
        raw = observe_window(row_id, 0, flattened, checkout=DEFAULT_CHECKOUT)
    except Exception as exc:  # governed: record, never crash the pilot
        failures.append({"row_id": row_id, "reason": f"{type(exc).__name__}: {exc}"})
        return None
    latencies.append(raw["runtime_seconds"])
    if shuffle_seed is not None:
        raw = shuffle_raw_steps(raw, seed=shuffle_seed)
    evidence = build_evidence(raw, TrajectoryRequest(case_id=row_id), "fabricpc", "0.3.2")
    if evidence.status != ObservationStatus.OBSERVED:
        failures.append({"row_id": row_id, "reason": evidence.reason})
        return None
    return evidence.to_dict()


def _fit_fabricpc_models(
    windows: List[np.ndarray],
    targets: List[float],
    row_ids: List[str],
    failures: List[Dict[str, Any]],
    latencies: List[float],
) -> Tuple[RidgeModel, RidgeModel]:
    terminal_features = []
    trajectory_features = []
    kept_targets_terminal = []
    kept_targets_trajectory = []
    total = len(windows)
    for i, (window, target, row_id) in enumerate(zip(windows, targets, row_ids)):
        evidence_terminal = _observe_evidence(
            f"{row_id}-terminal", window, True, failures, latencies
        )
        if evidence_terminal is not None:
            terminal_features.append(terminal_features_from_evidence(evidence_terminal))
            kept_targets_terminal.append(target)
        evidence_trajectory = _observe_evidence(
            f"{row_id}-trajectory", window, False, failures, latencies
        )
        if evidence_trajectory is not None:
            trajectory_features.append(trajectory_features_from_evidence(evidence_trajectory))
            kept_targets_trajectory.append(target)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  training FabricPC observations: {i + 1}/{total}")

    terminal_model = fit_ridge(terminal_features, kept_targets_terminal)
    trajectory_model = fit_ridge(trajectory_features, kept_targets_trajectory)
    return terminal_model, trajectory_model


def _make_fabricpc_predictor(
    model: RidgeModel,
    row_prefix: str,
    terminal_only: bool,
    shuffle: bool,
    failures: List[Dict[str, Any]],
    latencies: List[float],
    counter: List[int],
) -> Callable[[List[np.ndarray]], Optional[float]]:
    def _predict(window: List[np.ndarray]) -> Optional[float]:
        counter[0] += 1
        row_id = f"{row_prefix}-{counter[0]}"
        shuffle_seed = counter[0] if shuffle else None
        evidence = _observe_evidence(
            row_id, window, terminal_only, failures, latencies, shuffle_seed
        )
        if evidence is None:
            return None
        features = (
            terminal_features_from_evidence(evidence)
            if terminal_only
            else trajectory_features_from_evidence(evidence)
        )
        return predict_ridge(model, features)

    return _predict


def _route_disagreement_rate(reference: PolicyRunResult, arm: PolicyRunResult) -> float:
    total = len(reference.choices)
    if total == 0:
        return 0.0
    disagree = sum(1 for a, b in zip(reference.choices, arm.choices) if a != b)
    return disagree / total


def _residual_stats(records: List[ResidualCorrectionRecord]) -> Dict[str, float]:
    if not records:
        return {
            "mean_abs_applied_correction": 0.0,
            "clip_rate": 0.0,
            "failure_rate": 0.0,
            "zero_gate_rate": 0.0,
        }
    n = len(records)
    return {
        "mean_abs_applied_correction": statistics.mean(abs(r.applied_correction) for r in records),
        "clip_rate": sum(1 for r in records if r.status == "clipped") / n,
        "failure_rate": sum(1 for r in records if r.status == "failed") / n,
        "zero_gate_rate": sum(1 for r in records if r.status == "zero_gate") / n,
    }


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    sequences = generate_primary_dataset(seed=SEED, seeds_per_cell=1)
    train_sequences, test_sequences = _split_sequences(sequences)
    print(f"{len(train_sequences)} train cells, {len(test_sequences)} test cells")

    print("collecting training rows (oracle-compatible residual targets)...")
    train_windows, train_targets, train_row_ids = _collect_training_rows(train_sequences)
    print(f"{len(train_windows)} feasible training rows out of {len(train_sequences) * 12}")

    arm3_features = [
        flatten_window(w, MAX_WINDOW, RESIDUAL_CHANNEL_DIMENSION) for w in train_windows
    ]
    arm3_model = fit_ridge(arm3_features, train_targets)

    training_failures: List[Dict[str, Any]] = []
    training_latencies: List[float] = []
    print("fitting FabricPC terminal/trajectory residual models on training rows...")
    terminal_model, trajectory_model = _fit_fabricpc_models(
        train_windows, train_targets, train_row_ids, training_failures, training_latencies
    )

    hindsight: Dict[str, HindsightResult] = {
        s.sequence_id: compute_hindsight_optimum(s) for s in test_sequences
    }

    print("running arm 1 (no pricing) and arm 2 (frozen pacing)...")
    arm1_results = {}
    arm1_decisions = {}
    arm2_results = {}
    arm2_decisions = {}
    for seq in test_sequences:
        r1, d1 = simulate_policy(seq)
        r2, d2 = simulate_policy(seq, pricing_controller=_pacing(seq))
        arm1_results[seq.sequence_id] = r1
        arm1_decisions[seq.sequence_id] = d1
        arm2_results[seq.sequence_id] = r2
        arm2_decisions[seq.sequence_id] = d2

    fabricpc_failures: List[Dict[str, Any]] = []
    fabricpc_latencies: List[float] = []

    def _run_residual_arm(
        predictor_factory: Callable[[], Callable[[List[np.ndarray]], Optional[float]]],
        gate_fn: Callable[[Any, float], bool],
    ) -> Tuple[
        Dict[str, PolicyRunResult],
        Dict[str, List[PolicyDecision]],
        Dict[str, List[ResidualCorrectionRecord]],
    ]:
        results = {}
        decisions_by_seq = {}
        records_by_seq = {}
        for seq in test_sequences:
            controller = ResidualPricingController(
                base=_pacing(seq),
                predict_residual=predictor_factory(),
                max_correction_magnitude=MAX_CORRECTION_MAGNITUDE,
                gate_fn=gate_fn,
            )
            result, decisions = simulate_policy(seq, pricing_controller=controller)
            results[seq.sequence_id] = result
            decisions_by_seq[seq.sequence_id] = decisions
            records_by_seq[seq.sequence_id] = controller.records
        return results, decisions_by_seq, records_by_seq

    def _arm3_predictor() -> Callable[[List[np.ndarray]], Optional[float]]:
        def _predict(window: List[np.ndarray]) -> Optional[float]:
            features = flatten_window(window, MAX_WINDOW, RESIDUAL_CHANNEL_DIMENSION)
            return predict_ridge(arm3_model, features)

        return _predict

    print("running arm 3 (non-FabricPC windowed ridge)...")
    arm3_results, arm3_decisions, arm3_records = _run_residual_arm(_arm3_predictor, _scarcity_gate)

    print("running arm 4 (FabricPC terminal-state residual)...")
    counter4 = [0]
    arm4_results, arm4_decisions, arm4_records = _run_residual_arm(
        lambda: _make_fabricpc_predictor(
            terminal_model, "arm4", True, False, fabricpc_failures, fabricpc_latencies, counter4
        ),
        _scarcity_gate,
    )

    print("running arm 5 (FabricPC trajectory residual)...")
    counter5 = [0]
    arm5_results, arm5_decisions, arm5_records = _run_residual_arm(
        lambda: _make_fabricpc_predictor(
            trajectory_model, "arm5", False, False, fabricpc_failures, fabricpc_latencies, counter5
        ),
        _scarcity_gate,
    )

    print("running arm 6 (shuffled FabricPC trajectory, negative control)...")
    counter6 = [0]
    arm6_results, arm6_decisions, arm6_records = _run_residual_arm(
        lambda: _make_fabricpc_predictor(
            trajectory_model, "arm6", False, True, fabricpc_failures, fabricpc_latencies, counter6
        ),
        _scarcity_gate,
    )

    print("running arm 7 (FabricPC trajectory residual, gate forced open)...")
    counter7 = [0]
    arm7_results, arm7_decisions, arm7_records = _run_residual_arm(
        lambda: _make_fabricpc_predictor(
            trajectory_model, "arm7", False, False, fabricpc_failures, fabricpc_latencies, counter7
        ),
        _gate_always_open,
    )

    all_results = {
        "no_pricing": arm1_results,
        "frozen_pacing": arm2_results,
        "windowed_non_fabricpc": arm3_results,
        "fabricpc_terminal": arm4_results,
        "fabricpc_trajectory": arm5_results,
        "fabricpc_trajectory_shuffled": arm6_results,
        "fabricpc_trajectory_gate_open": arm7_results,
    }
    all_decisions = {
        "no_pricing": arm1_decisions,
        "frozen_pacing": arm2_decisions,
        "windowed_non_fabricpc": arm3_decisions,
        "fabricpc_terminal": arm4_decisions,
        "fabricpc_trajectory": arm5_decisions,
        "fabricpc_trajectory_shuffled": arm6_decisions,
        "fabricpc_trajectory_gate_open": arm7_decisions,
    }
    all_records = {
        "windowed_non_fabricpc": arm3_records,
        "fabricpc_terminal": arm4_records,
        "fabricpc_trajectory": arm5_records,
        "fabricpc_trajectory_shuffled": arm6_records,
        "fabricpc_trajectory_gate_open": arm7_records,
    }

    metrics = {
        name: regret_metrics(list(results.values()), hindsight)
        for name, results in all_results.items()
    }

    gate_reference_results = list(arm2_results.values())
    gate: Dict[str, Any] = {}
    for name in (
        "windowed_non_fabricpc",
        "fabricpc_terminal",
        "fabricpc_trajectory",
        "fabricpc_trajectory_shuffled",
        "fabricpc_trajectory_gate_open",
    ):
        arm_results = list(all_results[name].values())
        deltas_vs_pacing = paired_regret_deltas(arm_results, gate_reference_results, hindsight)
        ci_vs_pacing = bootstrap_ci(deltas_vs_pacing)
        gate[name] = {
            "paired_regret_delta_vs_frozen_pacing": ci_vs_pacing,
            "beats_frozen_pacing": ci_vs_pacing["ci_high"] < 0.0,
            "residual_stats": _residual_stats(
                [r for records in all_records[name].values() for r in records]
            ),
        }

    delta_arm5_vs_arm3 = paired_regret_deltas(
        list(arm5_results.values()), list(arm3_results.values()), hindsight
    )
    ci_arm5_vs_arm3 = bootstrap_ci(delta_arm5_vs_arm3)
    delta_arm5_vs_arm6 = paired_regret_deltas(
        list(arm5_results.values()), list(arm6_results.values()), hindsight
    )
    ci_arm5_vs_arm6 = bootstrap_ci(delta_arm5_vs_arm6)

    violations_arm5 = metrics["fabricpc_trajectory"]["total_violation_count"]
    violations_arm2 = metrics["frozen_pacing"]["total_violation_count"]

    activation_gate = {
        "criterion": (
            "arm 5 (FabricPC trajectory residual) must reduce regret vs frozen "
            "pacing AND vs the non-FabricPC windowed predictor AND be "
            "significantly better than the shuffled control, with no "
            "additional violations"
        ),
        "beats_frozen_pacing": gate["fabricpc_trajectory"]["beats_frozen_pacing"],
        "beats_non_fabricpc_windowed": {
            "paired_regret_delta": ci_arm5_vs_arm3,
            "passed": ci_arm5_vs_arm3["ci_high"] < 0.0,
        },
        "beats_shuffled_control": {
            "paired_regret_delta": ci_arm5_vs_arm6,
            "passed": ci_arm5_vs_arm6["ci_high"] < 0.0,
        },
        "no_additional_violations": violations_arm5 <= violations_arm2,
    }
    activation_gate["passed"] = bool(
        activation_gate["beats_frozen_pacing"]
        and activation_gate["beats_non_fabricpc_windowed"]["passed"]
        and activation_gate["beats_shuffled_control"]["passed"]
        and activation_gate["no_additional_violations"]
    )

    conservation_diagnostics: Dict[str, Any] = {}
    for name, decisions_by_seq in all_decisions.items():
        splits = [
            conservation_depletion_split(
                seq, decisions_by_seq[seq.sequence_id], hindsight[seq.sequence_id]
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
        }

    route_disagreement_vs_frozen_pacing = {}
    for name, results in all_results.items():
        if name in ("no_pricing", "frozen_pacing"):
            continue
        rates = [
            _route_disagreement_rate(arm2_results[sid], result) for sid, result in results.items()
        ]
        route_disagreement_vs_frozen_pacing[name] = statistics.mean(rates)

    all_latencies = training_latencies + fabricpc_latencies
    all_failures = training_failures + fabricpc_failures
    total_calls = len(train_windows) * 2 + sum(
        c[0] for c in (counter4, counter5, counter6, counter7)
    )

    def pct(values: List[float], q: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    report: Dict[str, Any] = {
        "schema": "compitum.fabricpc-residual-shadow-pilot-report/v1",
        "design": (
            "observation-only; frozen pacing (tranche 4/4.6) is the baseline "
            "every learned arm must beat, not the reactive controller; "
            "oracle-compatible residual targets built from the hindsight "
            "oracle, never fed to any online policy"
        ),
        "n_train_cells": len(train_sequences),
        "n_test_cells": len(test_sequences),
        "n_feasible_training_rows": len(train_windows),
        "arms": metrics,
        "gate_diagnostics": gate,
        "activation_gate": activation_gate,
        "conservation_depletion_diagnostics": conservation_diagnostics,
        "route_disagreement_vs_frozen_pacing": route_disagreement_vs_frozen_pacing,
        "latency_seconds": {
            "observe_p50": statistics.median(all_latencies) if all_latencies else float("nan"),
            "observe_p95": pct(all_latencies, 0.95),
            "observe_max": max(all_latencies) if all_latencies else float("nan"),
            "total_calls": total_calls,
            "governed_failure_rate": len(all_failures) / total_calls if total_calls else 0.0,
        },
        "governed_failures": all_failures,
        "honest_limitations": [
            "72-cell primary grid, 1 sequence per cell (opportunity_prevalence="
            "'rare' never consumes its RNG, per tranche 4.6 -- extra seeds "
            "would be duplicates, not independent draws, so none are used)",
            "48 train / 24 test cells, split by cell index, not by RNG seed",
            "ridge regression only, no hyperparameter search, matching every "
            "prior tranche's methodology",
            "regional-scope gate (lambda_base > 0.01) is a simple declared "
            "proxy, not derived from scenario metadata directly",
        ],
        "total_elapsed_seconds": time.perf_counter() - started,
    }

    out_path = ARTIFACTS / "residual_shadow_pilot_report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    out_path.write_text(rendered, encoding="utf-8", newline="")
    print(json.dumps(activation_gate, indent=2, default=str))
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
