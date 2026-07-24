"""Tranche 2 bounded, observation-only pilot (runs under .venv-fabricpc).

Pre-registered design (see docs/adr/0002-constraint-pressure-oracle.md and
the tranche 2 charter in the conversation history that authorized this
tranche):

Research question: given the currently observable Compitum state, does a
FabricPC inference trajectory improve held-out estimation of future
constraint activation or the value of constraint relaxation, beyond
ordinary static predictors and FabricPC terminal state alone? Tested
separately per constraint.

Arms (paired: identical cases/seeds/splits across all four):
  1. static  -- the declared 17-channel Compitum state + a one-hot
     constraint-index encoding (21 features total)
  2. +terminal   -- static + FabricPC terminal energy/grad/error (3 more)
  3. +trajectory -- static + FabricPC trajectory-summary features (8 more)
  4. +shuffled   -- static + trajectory features computed from a step
     -shuffled (temporally destroyed) version of the SAME real trajectory
  5. shadow_prices reference -- the frozen, real
     ReflectiveConstraintSolver.select() diagnostic, reported descriptively,
     never fit into the two-part model or treated as ground truth.

Activation criterion (pre-registered, not adjusted after seeing results):
arm 3 must beat BOTH arm 1 and arm 4 on held-out classification accuracy
AND held-out regression MAE. Sequence-level train/test split (never split
a sequence's own steps across train and test).

Routing/oracle computation never depends on FabricPC; FabricPC evidence is
collected alongside and never consulted for anything in this script.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for fabricpc_probe reuse if needed

from fabricpc_channel_observer import (  # noqa: E402
    DEFAULT_CHECKOUT,
    observe_channel_vector,
)

from compitum.constraint_oracle import (  # noqa: E402
    ConstraintPressureResult,
    SequenceStepResult,
    calibrate_threshold,
    classification_metrics,
    compute_constraint_pressure,
    compute_horizon_targets,
    compute_sequence_channels,
    fit_two_part_model,
    generate_controlled_dataset,
    predict_two_part,
    ranking_accuracy,
    regression_metrics,
    shuffle_raw_steps,
    terminal_features_from_evidence,
    trajectory_features_from_evidence,
)
from compitum.constraints import ReflectiveConstraintSolver  # noqa: E402
from compitum.control import LyapunovController  # noqa: E402
from compitum.trajectory.evidence import build_evidence  # noqa: E402
from compitum.trajectory.types import ObservationStatus, TrajectoryRequest  # noqa: E402

ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche2" / "artifacts"
MODEL_NAMES = ("current", "suppressed", "filler")
N_CONSTRAINTS = 4
HORIZON = 2
SEQUENCES_PER_SCENARIO = 4
STEPS_PER_SEQUENCE = 8
TRAIN_SEQUENCES_PER_SCENARIO = 2  # first N sequences of each scenario -> train


def _one_hot(index: int, size: int) -> List[float]:
    v = [0.0] * size
    v[index] = 1.0
    return v


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started_pilot = time.perf_counter()

    sequences = generate_controlled_dataset(
        seed=2026,
        sequences_per_scenario=SEQUENCES_PER_SCENARIO,
        steps_per_sequence=STEPS_PER_SEQUENCE,
    )

    rows_static: List[List[float]] = []
    rows_terminal: List[List[float]] = []
    rows_trajectory: List[List[float]] = []
    rows_shuffled: List[List[float]] = []
    rows_consequential: List[bool] = []
    rows_magnitude: List[Any] = []
    rows_shadow_price: List[float] = []
    rows_constraint_index: List[int] = []
    rows_is_train: List[bool] = []
    rows_case_key: List[Tuple[str, int]] = []
    rows_binding_within_horizon: List[bool] = []

    observe_latencies: List[float] = []
    governed_failures: List[Dict[str, Any]] = []
    total_calls = 0

    for seq in sequences:
        is_train_sequence = [True] * TRAIN_SEQUENCES_PER_SCENARIO + [False] * (
            SEQUENCES_PER_SCENARIO - TRAIN_SEQUENCES_PER_SCENARIO
        )
        # sequence_id encodes "<scenario>-<index:03d>"; recover the per
        # -scenario sequence index to decide train/test membership.
        seq_index = int(seq.sequence_id.rsplit("-", 1)[1])
        is_train = is_train_sequence[seq_index % SEQUENCES_PER_SCENARIO]

        pressures: List[ConstraintPressureResult] = []
        utilities_by_step = []
        shadow_prices_by_step = []
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities, context=case.context
            )
            pressures.append(pressure)
            utilities_by_step.append(case.utilities)
            solver = ReflectiveConstraintSolver(seq.A, seq.b)
            _, info = solver.select(case.xB, seq.models, case.utilities, context=case.context)
            shadow_prices_by_step.append(
                [info["shadow_prices"].get(f"lambda_{i}", 0.0) for i in range(N_CONSTRAINTS)]
            )

        controller = LyapunovController()
        channel_vectors = compute_sequence_channels(
            pressures, utilities_by_step, MODEL_NAMES, controller=controller
        )

        step_results = [SequenceStepResult(step=t, pressure=p) for t, p in enumerate(pressures)]
        horizon_results = compute_horizon_targets(step_results, horizon=HORIZON)

        for t, (pressure, vector) in enumerate(zip(pressures, channel_vectors)):
            case_id = f"{seq.sequence_id}-step{t}"
            request = TrajectoryRequest(case_id=case_id, seeds={"step": t})
            total_calls += 1
            try:
                raw = observe_channel_vector(seq.sequence_id, t, vector, checkout=DEFAULT_CHECKOUT)
            except Exception as exc:  # governed: record, never crash the pilot
                governed_failures.append(
                    {"case_id": case_id, "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue
            observe_latencies.append(raw["runtime_seconds"])
            evidence = build_evidence(raw, request, "fabricpc", "0.3.2")
            if evidence.status != ObservationStatus.OBSERVED:
                governed_failures.append({"case_id": case_id, "reason": evidence.reason})
                continue

            shuffled_raw = shuffle_raw_steps(raw, seed=t)
            shuffled_evidence = build_evidence(
                shuffled_raw, TrajectoryRequest(case_id=f"{case_id}-shuffled"), "fabricpc", "0.3.2"
            )

            evidence_dict = evidence.to_dict()
            shuffled_dict = shuffled_evidence.to_dict()
            terminal_feats = terminal_features_from_evidence(evidence_dict)
            trajectory_feats = trajectory_features_from_evidence(evidence_dict)
            shuffled_feats = trajectory_features_from_evidence(shuffled_dict)

            for i in range(N_CONSTRAINTS):
                target = pressure.targets[i]
                static_feats = list(vector) + _one_hot(i, N_CONSTRAINTS)
                rows_static.append(static_feats)
                rows_terminal.append(static_feats + terminal_feats)
                rows_trajectory.append(static_feats + trajectory_feats)
                rows_shuffled.append(static_feats + shuffled_feats)
                consequential = target.reason == "recovers_feasibility"
                rows_consequential.append(consequential)
                rows_magnitude.append(target.critical_relaxation if consequential else None)
                rows_shadow_price.append(shadow_prices_by_step[t][i])
                rows_constraint_index.append(i)
                rows_is_train.append(is_train)
                rows_case_key.append((seq.sequence_id, t))
                rows_binding_within_horizon.append(horizon_results[t][i].binding_within_horizon)

    print(f"Assembled {len(rows_static)} (case, constraint) rows from {total_calls} observations")
    print(f"Governed failures: {len(governed_failures)}")

    arms = {
        "static": np.array(rows_static),
        "terminal": np.array(rows_terminal),
        "trajectory": np.array(rows_trajectory),
        "shuffled": np.array(rows_shuffled),
    }
    is_train = np.array(rows_is_train)
    consequential = rows_consequential
    magnitude = rows_magnitude
    constraint_index = np.array(rows_constraint_index)

    report: Dict[str, Any] = {
        "schema": "compitum.constraint-pressure-pilot-report/v1",
        "design": (
            "observation-only; FabricPC evidence never consulted for routing "
            "or by the oracle; sequence-level train/test split"
        ),
        "n_rows": len(rows_static),
        "n_train_rows": int(is_train.sum()),
        "n_test_rows": int((~is_train).sum()),
        "governed_failures": governed_failures,
        "arms": {},
    }

    arm_test_predictions: Dict[str, np.ndarray] = {}
    for arm_name, X in arms.items():
        X_train, X_test = X[is_train], X[~is_train]
        cons_train = [c for c, keep in zip(consequential, is_train) if keep]
        mag_train = [m for m, keep in zip(magnitude, is_train) if keep]
        cons_test = [c for c, keep in zip(consequential, ~is_train) if keep]
        mag_test = [m for m, keep in zip(magnitude, ~is_train) if keep]

        model = fit_two_part_model(X_train, cons_train, mag_train)
        p_pred, mag_pred = predict_two_part(model, X_test)
        arm_test_predictions[arm_name] = mag_pred

        # Threshold is calibrated from TRAINING predictions/labels only (never
        # test labels) -- a fixed 0.5 cutoff is inappropriate here since the
        # positive ("consequential") rate is far below 50% and the raw ridge
        # score regresses toward the mean, so it may never cross 0.5 even
        # when it carries real separating signal.
        p_train, _ = predict_two_part(model, X_train)
        threshold = calibrate_threshold(p_train, cons_train)

        overall_cls = classification_metrics(cons_test, p_pred, threshold=threshold)
        cons_rows = [i for i, c in enumerate(cons_test) if c]
        overall_reg = regression_metrics(
            [mag_test[i] for i in cons_rows], [mag_pred[i] for i in cons_rows]
        )

        per_constraint: Dict[str, Any] = {}
        test_constraint_index = constraint_index[~is_train]
        for i in range(N_CONSTRAINTS):
            mask = test_constraint_index == i
            idxs = np.nonzero(mask)[0]
            c_true_i = [cons_test[j] for j in idxs]
            p_pred_i = [p_pred[j] for j in idxs]
            cls_i = classification_metrics(c_true_i, p_pred_i, threshold=threshold)
            cons_idxs_i = [j for j in idxs if cons_test[j]]
            reg_i = regression_metrics(
                [mag_test[j] for j in cons_idxs_i], [mag_pred[j] for j in cons_idxs_i]
            )
            per_constraint[str(i)] = {"classification": cls_i, "regression": reg_i}

        report["arms"][arm_name] = {
            "classification": overall_cls,
            "regression": overall_reg,
            "per_constraint": per_constraint,
            "calibrated_threshold": threshold,
        }

    # Ranking accuracy per arm: group held-out rows by (sequence_id, step).
    test_case_keys = [k for k, keep in zip(rows_case_key, is_train) if not keep]
    unique_cases = sorted(set(test_case_keys), key=lambda k: (k[0], k[1]))
    test_mag_true = [m for m, keep in zip(magnitude, is_train) if not keep]
    test_cons_true = [c for c, keep in zip(consequential, is_train) if not keep]
    test_constraint_idx_list = [i for i, keep in zip(rows_constraint_index, is_train) if not keep]
    for arm_name, mag_pred in arm_test_predictions.items():
        cases_grouped: List[List[Tuple[float, float]]] = []
        for case_key in unique_cases:
            # Collect (constraint_index, true_significance, pred_significance)
            # for this case, then sort by constraint_index so ranking_accuracy
            # compares corresponding constraints across the true/pred pairing
            # (the absolute order doesn't affect argmax agreement, but keeping
            # it deterministic avoids depending on dict/list iteration order).
            triples = []
            for j, key in enumerate(test_case_keys):
                if key != case_key:
                    continue
                true_sig = -test_mag_true[j] if test_cons_true[j] else -math.inf
                pred_sig = -float(mag_pred[j])
                triples.append((test_constraint_idx_list[j], true_sig, pred_sig))
            triples.sort(key=lambda t: t[0])
            pairs = [(t[1], t[2]) for t in triples]
            if len(pairs) == N_CONSTRAINTS:
                cases_grouped.append(pairs)
        report["arms"][arm_name]["ranking_accuracy"] = ranking_accuracy(cases_grouped)

    # Horizon precision/recall using each arm's P(consequential) as a proxy
    # for "will this constraint bind within HORIZON steps".
    test_binding = [b for b, keep in zip(rows_binding_within_horizon, is_train) if not keep]
    train_binding = [b for b, keep in zip(rows_binding_within_horizon, is_train) if keep]
    for arm_name, X in arms.items():
        X_train, X_test = X[is_train], X[~is_train]
        model = fit_two_part_model(
            X_train,
            [c for c, keep in zip(consequential, is_train) if keep],
            [m for m, keep in zip(magnitude, is_train) if keep],
        )
        p_pred, _ = predict_two_part(model, X_test)
        # This reuses the P(consequential) classifier as a proxy score for a
        # different label (binding within HORIZON steps), so its own
        # training-derived threshold (calibrated against train_binding, not
        # cons_train) is recalculated here rather than reusing the earlier one.
        p_pred_train, _ = predict_two_part(model, X_train)
        horizon_threshold = calibrate_threshold(p_pred_train, train_binding)
        report["arms"][arm_name]["binding_within_horizon"] = classification_metrics(
            test_binding, p_pred, threshold=horizon_threshold
        )

    # Shadow-price reference (descriptive only, never fit/treated as truth).
    test_shadow = [s for s, keep in zip(rows_shadow_price, is_train) if not keep]
    nonzero_shadow = [abs(s) > 1e-9 for s in test_shadow]
    report["shadow_price_reference"] = classification_metrics(
        test_cons_true, [1.0 if nz else 0.0 for nz in nonzero_shadow]
    )
    report["shadow_price_reference"]["note"] = (
        "descriptive agreement between the frozen finite-difference "
        "shadow_prices diagnostic (nonzero vs zero) and the oracle's own "
        "consequential label -- reference only, never fit or treated as "
        "ground truth"
    )

    # Pre-registered activation gate.
    def _beats(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        acc_ok = a["classification"]["accuracy"] > b["classification"]["accuracy"]
        mae_a, mae_b = a["regression"]["mae"], b["regression"]["mae"]
        mae_ok = (not math.isnan(mae_a) and not math.isnan(mae_b) and mae_a < mae_b) or (
            math.isnan(mae_a) and math.isnan(mae_b)
        )
        return bool(acc_ok and mae_ok)

    gate_vs_baseline = _beats(report["arms"]["trajectory"], report["arms"]["static"])
    gate_vs_shuffled = _beats(report["arms"]["trajectory"], report["arms"]["shuffled"])
    report["activation_gate"] = {
        "criterion": (
            "trajectory arm must beat BOTH static baseline AND shuffled "
            "control on held-out accuracy AND MAE"
        ),
        "beats_static_baseline": gate_vs_baseline,
        "beats_shuffled_control": gate_vs_shuffled,
        "passed": bool(gate_vs_baseline and gate_vs_shuffled),
    }

    def pct(values: List[float], q: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    report["latency_seconds"] = {
        "observe_p50": statistics.median(observe_latencies) if observe_latencies else float("nan"),
        "observe_p95": pct(observe_latencies, 0.95),
        "observe_max": max(observe_latencies) if observe_latencies else float("nan"),
        "total_calls": total_calls,
        "governed_failure_rate": len(governed_failures) / total_calls if total_calls else 0.0,
    }
    report["honest_limitations"] = [
        f"{SEQUENCES_PER_SCENARIO} sequences per scenario ({TRAIN_SEQUENCES_PER_SCENARIO} train / "
        f"{SEQUENCES_PER_SCENARIO - TRAIN_SEQUENCES_PER_SCENARIO} test), "
        f"{STEPS_PER_SEQUENCE} steps each: a bounded pipeline-validation pilot, not a "
        "high-powered study",
        "controlled synthetic scenarios only -- no realized routing labels in this tranche",
        (
            "two-part model is a simple ridge fit (no hyperparameter search), "
            "matching tranche 1's own methodology"
        ),
        (
            "targets are the independently-derived oracle's own labels, "
            "not externally realized outcomes"
        ),
    ]
    report["total_elapsed_seconds"] = time.perf_counter() - started_pilot

    out_path = ARTIFACTS / "pilot_report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    out_path.write_text(rendered, encoding="utf-8", newline="")
    print(json.dumps(report["activation_gate"], indent=2))
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
