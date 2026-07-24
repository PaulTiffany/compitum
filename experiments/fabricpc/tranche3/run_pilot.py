"""Tranche 3 bounded, observation-only pilot (runs under .venv-fabricpc).

Pre-registered design (see docs/adr/0003-dynamic-constraint-regret.md):

Research question: in a sequential routing environment where different
model choices consume different constrained resources and therefore change
future feasibility, does FabricPC trajectory observation improve an
otherwise-valid online dual-pricing policy's held-out cumulative
constrained regret -- beyond a no-predictor dual baseline and a simple
non-FabricPC sequential (EWMA) predictor?

Training regime: on TRAINING sequences only, run one reference rollout per
sequence (the online dual controller + an EWMA forecaster that learns
online, i.e. "arm 3 deployed during a training period"). This rollout
produces, per step: (a) EWMA's own online bias update (letting it learn
naturally), and (b) FabricPC trajectory-feature observations of that
step's channel vector, paired against the FULL per-(model,resource) ground
-truth residual (realized - expected consumption) for every candidate --
not just the model actually chosen -- since this is an offline supervised
fit, analogous to tranche 2's two-part model methodology (fit on train,
frozen and applied to test, never touching test-sequence ground truth).
A small ridge regression per (model, resource) pair is fit on these
examples. EWMA is then frozen (no further .update calls) before any
test-sequence simulation.

Five paired arms per held-out test sequence (identical sequence, same
hindsight target):
  1. static/frozen pricing (no dual controller, greedy on base utility)
  2. online dual, no learned predictor
  3. online dual + frozen EWMA (non-FabricPC sequential baseline)
  4. online dual + FabricPC trajectory-pressure prediction
  5. arm 4 with a shuffled/temporally-destroyed trajectory (negative control)

Gate (pre-registered, not adjusted after seeing results): on held-out test
sequences, arm 4 must show a paired bootstrap-CI-significant reduction in
mean cumulative constrained regret vs BOTH arm 2 and arm 3, must not
increase total hard violations relative to either, and must be
significantly better than arm 5. Routing/oracle computation never depends
on FabricPC; FabricPC evidence is collected alongside and never consulted
by the hindsight oracle or the dataset generator.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fabricpc_regret_observer import DEFAULT_CHECKOUT, observe_channel_vector  # noqa: E402

from compitum.constraint_oracle.experiment import (  # noqa: E402
    shuffle_raw_steps,
    trajectory_features_from_evidence,
)
from compitum.regret_lab import (  # noqa: E402
    MODEL_NAMES,
    RESOURCE_NAMES,
    DualController,
    DynamicSequence,
    EWMAForecaster,
    ForecastContext,
    PolicyRunResult,
    bootstrap_ci,
    compute_hindsight_optimum,
    compute_regret_channel_vector,
    generate_dynamic_dataset,
    paired_regret_deltas,
    regret_metrics,
    simulate_policy,
)
from compitum.trajectory.evidence import build_evidence  # noqa: E402
from compitum.trajectory.types import ObservationStatus, TrajectoryRequest  # noqa: E402

ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche3" / "artifacts"
SEQUENCES_PER_SCENARIO = 4
TRAIN_SEQUENCES_PER_SCENARIO = 2
STEPS_PER_SEQUENCE = 8
DUAL_ETA = 0.5
DUAL_LAMBDA_MAX = 20.0
EWMA_ALPHA = 0.3
RIDGE_LAMBDA = 1.0

RidgeModel = Tuple[np.ndarray, float, np.ndarray, np.ndarray]  # (weights, y_mean, mean, scale)


def _ridge_fit(
    features: np.ndarray, targets: np.ndarray, ridge: float = RIDGE_LAMBDA
) -> RidgeModel:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.0] = 1.0
    standardized = (features - mean) / scale
    y_mean = float(targets.mean())
    gram = standardized.T @ standardized + ridge * np.eye(standardized.shape[1])
    weights = np.linalg.solve(gram, standardized.T @ (targets - y_mean))
    return weights, y_mean, mean, scale


def _ridge_predict(model: RidgeModel, features: np.ndarray) -> float:
    weights, y_mean, mean, scale = model
    standardized = (features - mean) / scale
    return float(standardized @ weights + y_mean)


class FabricPCForecaster:
    """Observation-only: predicts a per-(model, resource) consumption
    correction from FabricPC trajectory features of the current step's
    declared channel vector. On any observer failure, degrades gracefully
    to the unmodified forecast (recorded as a governed failure), never
    crashes the simulation."""

    def __init__(
        self,
        ridge_models: Dict[Tuple[str, str], RidgeModel],
        sequence_id: str,
        shuffle: bool = False,
    ) -> None:
        self.ridge_models = ridge_models
        self.sequence_id = sequence_id
        self.shuffle = shuffle
        self.failures: List[Dict[str, Any]] = []
        self.latencies: List[float] = []
        self.calls = 0

    def __call__(
        self, expected_consumption: Dict[str, Dict[str, float]], context: ForecastContext
    ) -> Dict[str, Dict[str, float]]:
        self.calls += 1
        vector = compute_regret_channel_vector(
            context.remaining,
            context.case,
            context.lambda_price,
            context.steps_left,
            context.total_steps,
        )
        step = context.case.step
        try:
            raw = observe_channel_vector(self.sequence_id, step, vector, checkout=DEFAULT_CHECKOUT)
        except Exception as exc:  # governed: record, never crash the pilot
            self.failures.append(
                {
                    "sequence_id": self.sequence_id,
                    "step": step,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            return expected_consumption
        self.latencies.append(raw["runtime_seconds"])
        if self.shuffle:
            raw = shuffle_raw_steps(raw, seed=step)
        evidence = build_evidence(
            raw, TrajectoryRequest(case_id=f"{self.sequence_id}-step{step}"), "fabricpc", "0.3.2"
        )
        if evidence.status != ObservationStatus.OBSERVED:
            self.failures.append(
                {"sequence_id": self.sequence_id, "step": step, "reason": evidence.reason}
            )
            return expected_consumption
        features = np.array(trajectory_features_from_evidence(evidence.to_dict()))
        predicted: Dict[str, Dict[str, float]] = {}
        for m in MODEL_NAMES:
            predicted[m] = {}
            for r in RESOURCE_NAMES:
                residual = _ridge_predict(self.ridge_models[(m, r)], features)
                predicted[m][r] = expected_consumption[m][r] + residual
        return predicted


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


def _train(
    train_sequences: List[DynamicSequence],
) -> Tuple[EWMAForecaster, Dict[Tuple[str, str], RidgeModel], List[Dict[str, Any]], List[float]]:
    ewma = EWMAForecaster(alpha=EWMA_ALPHA)
    features_by_pair: Dict[Tuple[str, str], List[np.ndarray]] = {
        (m, r): [] for m in MODEL_NAMES for r in RESOURCE_NAMES
    }
    targets_by_pair: Dict[Tuple[str, str], List[float]] = {
        (m, r): [] for m in MODEL_NAMES for r in RESOURCE_NAMES
    }
    failures: List[Dict[str, Any]] = []
    latencies: List[float] = []
    total_steps = sum(len(seq.cases) for seq in train_sequences)
    done = 0
    started = time.perf_counter()

    for seq in train_sequences:
        controller = DualController(
            resource_names=RESOURCE_NAMES, eta=DUAL_ETA, lambda_max=DUAL_LAMBDA_MAX
        )
        _, decisions = simulate_policy(
            seq, dual_controller=controller, forecaster=ewma, forecaster_update=ewma.update
        )
        for t, case in enumerate(seq.cases):
            decision = decisions[t]
            vector = compute_regret_channel_vector(
                decision.remaining_before,
                case,
                decision.lambda_price_before,
                len(seq.cases) - t,
                len(seq.cases),
            )
            try:
                raw = observe_channel_vector(seq.sequence_id, t, vector, checkout=DEFAULT_CHECKOUT)
            except Exception as exc:  # governed: record, skip this training row
                failures.append(
                    {
                        "sequence_id": seq.sequence_id,
                        "step": t,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                done += 1
                continue
            latencies.append(raw["runtime_seconds"])
            evidence = build_evidence(
                raw, TrajectoryRequest(case_id=f"{seq.sequence_id}-step{t}"), "fabricpc", "0.3.2"
            )
            if evidence.status == ObservationStatus.OBSERVED:
                features = np.array(trajectory_features_from_evidence(evidence.to_dict()))
                for m in MODEL_NAMES:
                    for r in RESOURCE_NAMES:
                        residual = case.realized_consumption[m][r] - case.expected_consumption[m][r]
                        features_by_pair[(m, r)].append(features)
                        targets_by_pair[(m, r)].append(residual)
            else:
                failures.append(
                    {"sequence_id": seq.sequence_id, "step": t, "reason": evidence.reason}
                )
            done += 1
            if done % 20 == 0 or done == total_steps:
                elapsed = time.perf_counter() - started
                print(f"  training observations: {done}/{total_steps} ({elapsed:.0f}s elapsed)")

    ridge_models: Dict[Tuple[str, str], RidgeModel] = {}
    for key in features_by_pair:
        X = np.array(features_by_pair[key])
        y = np.array(targets_by_pair[key])
        ridge_models[key] = _ridge_fit(X, y)

    return ewma, ridge_models, failures, latencies


def _run_arm(
    test_sequences: List[DynamicSequence],
    build_controller: Optional[Any],
    forecaster: Optional[Any] = None,
    forecaster_update: Optional[Any] = None,
) -> List[PolicyRunResult]:
    results = []
    for seq in test_sequences:
        controller = build_controller() if build_controller is not None else None
        result, _ = simulate_policy(
            seq,
            dual_controller=controller,
            forecaster=forecaster,
            forecaster_update=forecaster_update,
        )
        results.append(result)
    return results


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started_pilot = time.perf_counter()

    sequences = generate_dynamic_dataset(
        seed=2026,
        sequences_per_scenario=SEQUENCES_PER_SCENARIO,
        steps_per_sequence=STEPS_PER_SEQUENCE,
    )
    train_sequences, test_sequences = _split_sequences(sequences)
    print(f"{len(train_sequences)} train sequences, {len(test_sequences)} test sequences")

    print("training EWMA + FabricPC ridge models on training sequences...")
    ewma, ridge_models, train_failures, train_latencies = _train(train_sequences)

    hindsight = {seq.sequence_id: compute_hindsight_optimum(seq) for seq in test_sequences}

    print("running arm 1 (static/frozen)...")
    arm1 = _run_arm(test_sequences, build_controller=None)

    def _dual() -> DualController:
        return DualController(
            resource_names=RESOURCE_NAMES, eta=DUAL_ETA, lambda_max=DUAL_LAMBDA_MAX
        )

    print("running arm 2 (dual, no predictor)...")
    arm2 = _run_arm(test_sequences, build_controller=_dual)

    print("running arm 3 (dual + frozen EWMA)...")
    arm3 = _run_arm(test_sequences, build_controller=_dual, forecaster=ewma, forecaster_update=None)

    print("running arm 4 (dual + FabricPC)...")
    fabricpc_forecasters = []
    arm4 = []
    for seq in test_sequences:
        forecaster = FabricPCForecaster(ridge_models, seq.sequence_id, shuffle=False)
        fabricpc_forecasters.append(forecaster)
        result, _ = simulate_policy(seq, dual_controller=_dual(), forecaster=forecaster)
        arm4.append(result)

    print("running arm 5 (dual + FabricPC, shuffled control)...")
    shuffled_forecasters = []
    arm5 = []
    for seq in test_sequences:
        forecaster = FabricPCForecaster(ridge_models, seq.sequence_id, shuffle=True)
        shuffled_forecasters.append(forecaster)
        result, _ = simulate_policy(seq, dual_controller=_dual(), forecaster=forecaster)
        arm5.append(result)

    arms = {
        "static": arm1,
        "dual_no_predictor": arm2,
        "dual_ewma": arm3,
        "dual_fabricpc": arm4,
        "dual_fabricpc_shuffled": arm5,
    }
    metrics = {name: regret_metrics(results, hindsight) for name, results in arms.items()}

    delta_vs_arm2 = paired_regret_deltas(arm4, arm2, hindsight)
    delta_vs_arm3 = paired_regret_deltas(arm4, arm3, hindsight)
    delta_vs_arm5 = paired_regret_deltas(arm4, arm5, hindsight)
    ci_vs_arm2 = bootstrap_ci(delta_vs_arm2)
    ci_vs_arm3 = bootstrap_ci(delta_vs_arm3)
    ci_vs_arm5 = bootstrap_ci(delta_vs_arm5)

    beats_arm2 = ci_vs_arm2["ci_high"] < 0.0
    beats_arm3 = ci_vs_arm3["ci_high"] < 0.0
    distinguishable_from_arm5 = ci_vs_arm5["ci_high"] < 0.0
    violations_not_increased = metrics["dual_fabricpc"]["total_violation_count"] <= max(
        metrics["dual_no_predictor"]["total_violation_count"],
        metrics["dual_ewma"]["total_violation_count"],
    )
    passed = bool(
        beats_arm2 and beats_arm3 and distinguishable_from_arm5 and violations_not_increased
    )

    fabricpc_failures = [
        f for fc in fabricpc_forecasters + shuffled_forecasters for f in fc.failures
    ]
    fabricpc_latencies = [
        latency for fc in fabricpc_forecasters + shuffled_forecasters for latency in fc.latencies
    ]
    total_fabricpc_calls = sum(fc.calls for fc in fabricpc_forecasters + shuffled_forecasters)
    all_latencies = train_latencies + fabricpc_latencies
    all_failures = train_failures + fabricpc_failures
    total_calls = sum(len(seq.cases) for seq in train_sequences) + total_fabricpc_calls

    def pct(values: List[float], q: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    report: Dict[str, Any] = {
        "schema": "compitum.regret-pilot-report/v1",
        "design": (
            "observation-only; hindsight oracle and dataset never consult FabricPC; "
            "sequence-level train/test split; FabricPC ridge models and EWMA fit "
            "offline on training sequences, then frozen for all test-time arms"
        ),
        "n_train_sequences": len(train_sequences),
        "n_test_sequences": len(test_sequences),
        "arms": metrics,
        "activation_gate": {
            "criterion": (
                "arm 4 (dual+FabricPC) must show a paired bootstrap-CI-significant "
                "reduction in mean regret vs BOTH arm 2 (dual, no predictor) and arm 3 "
                "(dual+EWMA), must not increase total violations vs either, and must "
                "be significantly better than arm 5 (shuffled control)"
            ),
            "paired_regret_delta_vs_dual_no_predictor": ci_vs_arm2,
            "paired_regret_delta_vs_dual_ewma": ci_vs_arm3,
            "paired_regret_delta_vs_shuffled_control": ci_vs_arm5,
            "beats_dual_no_predictor": beats_arm2,
            "beats_dual_ewma": beats_arm3,
            "distinguishable_from_shuffled_control": distinguishable_from_arm5,
            "violations_not_increased": violations_not_increased,
            "passed": passed,
        },
        "governed_failures": all_failures,
        "latency_seconds": {
            "observe_p50": statistics.median(all_latencies) if all_latencies else float("nan"),
            "observe_p95": pct(all_latencies, 0.95),
            "observe_max": max(all_latencies) if all_latencies else float("nan"),
            "total_calls": total_calls,
            "governed_failure_rate": len(all_failures) / total_calls if total_calls else 0.0,
        },
        "honest_limitations": [
            f"{SEQUENCES_PER_SCENARIO} sequences per scenario "
            f"({TRAIN_SEQUENCES_PER_SCENARIO} train / "
            f"{SEQUENCES_PER_SCENARIO - TRAIN_SEQUENCES_PER_SCENARIO} "
            f"test), {STEPS_PER_SEQUENCE} steps each: a bounded pipeline-validation pilot, not a "
            "high-powered study",
            "FabricPC ridge models are trained on channel-vector trajectories induced by an "
            "EWMA-forecaster reference rollout on training sequences, not on trajectories "
            "induced by their own eventual test-time deployment policy -- a legitimate but "
            "imperfect offline-training setup, consistent with tranche 2's own methodology",
            "FabricPC 'trajectory' means the PC graph's own settling dynamics within one inference "
            "call (as in tranches 1-2), not a multi-environment-step window; testing whether "
            "windowing several past environment steps into FabricPC's own input helps is untested",
            "ridge regression per (model, resource) pair, no hyperparameter search, matching "
            "tranches 1-2's own methodology",
            "controlled synthetic scenarios only -- no realized routing labels in this tranche",
        ],
        "total_elapsed_seconds": time.perf_counter() - started_pilot,
    }

    out_path = ARTIFACTS / "pilot_report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    out_path.write_text(rendered, encoding="utf-8", newline="")
    print(json.dumps(report["activation_gate"], indent=2))
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
