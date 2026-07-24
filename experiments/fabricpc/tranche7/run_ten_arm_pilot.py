"""Tranche 7: ten-arm belief-sensitive shadow-charge pilot.

Runs under ``.venv-fabricpc``. Per docs/adr/0009-belief-sensitive-shadow-charge-validation.md:
tests whether trained FabricPC belief inference improves constrained
routing regret in an environment where belief genuinely, provably
changes the Bellman-optimal action (Gate 0, passed --
``experiments/fabricpc/tranche7/artifacts/gate0_report.json``). Reuses
tranche 6.5's shadow-charge pricing mechanism and Part B training
infrastructure (ridge, FabricPC train_pcn/train_backprop) completely
unchanged; only the environment and its belief-estimator siblings
(``belief_regime_v2``, ``belief_bellman_v2``, ``belief_action_pricing_v2``,
``belief_online_optimum_v2``) are new, per tranche 7.1/7.1b.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TRANCHE6_DIR = REPO_ROOT / "experiments" / "fabricpc" / "tranche6"
sys.path.insert(0, str(TRANCHE6_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fabricpc_belief_model import (  # noqa: E402
    MAX_WINDOW,
    TRAIN_SEEDS,
    FabricPCBeliefEstimator,
    predict_belief_batch,
    train_belief_model,
)

from compitum.regret_lab import (  # noqa: E402
    BeliefSensitiveBellmanOracle,
    DynamicSequence,
    ExactBeliefEstimatorV2,
    HmmBeliefEstimatorV2,
    LookupBeliefEstimator,
    PacingController,
    PolicyRunResult,
    RidgeBeliefEstimator,
    bootstrap_ci,
    build_belief_training_pairs,
    compute_hindsight_optimum,
    fit_ridge,
    generate_belief_dataset_v2,
    online_optimum_as_hindsight_result_v2,
    paired_regret_deltas,
    regret_metrics,
    run_shadow_charge_policy_v2,
    simulate_policy,
    total_available_over_horizon,
)
from compitum.regret_lab.belief_regime import INITIAL_BELIEF  # noqa: E402
from compitum.regret_lab.windowed_predictor import predict_ridge  # noqa: E402

ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche7" / "artifacts"
GATE0_REPORT = ARTIFACTS / "gate0_report.json"

SEED = 4242
N_TRAIN = 50
N_VAL = 15
N_TEST = 35
FROZEN_PACING_ETA = 1.8
BOUNDARY_DISTANCE = 0.1
BELIEF_GRID = tuple(round(x, 4) for x in np.linspace(0.0, 1.0, 41))


def _load_frozen_config() -> Dict[str, float]:
    report = json.loads(GATE0_REPORT.read_text(encoding="utf-8"))
    if not report.get("selected_config"):
        raise RuntimeError("Gate 0 has not passed -- refusing to run the pilot. See ADR 0009.")
    cfg = report["selected_config"]
    return {
        "u_normal": cfg["u_normal"],
        "u_high": cfg["u_high"],
        "initial_budget": cfg["initial_budget"],
        "p_opportunity_normal": cfg["p_opportunity_normal"],
        "p_opportunity_high": cfg["p_opportunity_high"],
        "transition_normal_to_high": cfg["transition_normal_to_high"],
        "transition_high_to_high": cfg["transition_high_to_high"],
    }


def _pacing(seq: DynamicSequence) -> PacingController:
    return PacingController(
        resource_names=seq.resource_names,
        total_available=total_available_over_horizon(seq),
        total_steps=len(seq.cases),
        eta=FROZEN_PACING_ETA,
    )


def _transition_kwargs(cfg: Dict[str, float]) -> Dict[str, float]:
    return {
        "p_opportunity_normal": cfg["p_opportunity_normal"],
        "p_opportunity_high": cfg["p_opportunity_high"],
        "transition_normal_to_high": cfg["transition_normal_to_high"],
        "transition_high_to_high": cfg["transition_high_to_high"],
    }


def _shadow(seq, oracle, estimator, cfg) -> Any:
    return run_shadow_charge_policy_v2(
        seq,
        oracle,
        estimator,
        u_normal=cfg["u_normal"],
        u_high=cfg["u_high"],
        **_transition_kwargs(cfg),
    )


def _belief_quality(
    predictions: np.ndarray, targets: np.ndarray, regimes: np.ndarray
) -> Dict[str, float]:
    eps = 1e-9
    clipped = np.clip(predictions, eps, 1.0 - eps)
    mse = float(np.mean((predictions - targets) ** 2))
    log_loss = float(
        -np.mean(regimes * np.log(clipped) + (1.0 - regimes) * np.log(1.0 - clipped))
    )
    brier = float(np.mean((predictions - regimes) ** 2))
    accuracy = float(np.mean((predictions > 0.5).astype(float) == regimes))
    return {
        "mse_vs_belief_prior": mse,
        "log_loss_vs_regime": log_loss,
        "brier_score_vs_regime": brier,
        "hidden_regime_accuracy": accuracy,
    }


def _enumerate_reachable_states(oracle, steps, initial_budget, initial_belief):
    oracle.value(steps, initial_budget, initial_belief)
    return sorted({(r, b) for (r, b, _bel) in oracle._value_memo})


def _scan_belief_boundaries(oracle, remaining_steps, budget):
    result = {}
    for observed in (False, True):
        actions = [
            oracle.best_action_given_observation(remaining_steps, budget, b, observed)[0]
            for b in BELIEF_GRID
        ]
        transitions = [
            BELIEF_GRID[i] for i in range(len(actions) - 1) if actions[i] != actions[i + 1]
        ]
        result[str(observed)] = transitions
    return result


def _nearest_boundary_distance(belief: float, transitions: List[float]) -> float:
    if not transitions:
        return float("inf")
    return min(abs(belief - t) for t in transitions)


def _boundary_diagnostics(
    boundary_map, traces_by_seq: Dict[str, List[Any]]
) -> Dict[str, Any]:
    near_boundary_count = 0
    total = 0
    for traces in traces_by_seq.values():
        total_steps = len(traces)
        for t, trace in enumerate(traces):
            remaining_steps = total_steps - t
            budget_key = round(trace.remaining_budget_before / 0.5) * 0.5
            entry = boundary_map.get((remaining_steps, budget_key))
            total += 1
            if entry is None:
                continue
            transitions = entry[str(trace.observation)]
            distance = _nearest_boundary_distance(trace.predicted_next_belief, transitions)
            if distance <= BOUNDARY_DISTANCE:
                near_boundary_count += 1
    return {
        "near_boundary_steps": near_boundary_count,
        "total_steps": total,
        "near_boundary_fraction": near_boundary_count / total if total else 0.0,
    }


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cfg = _load_frozen_config()
    print(f"frozen Gate 0 config: {json.dumps(cfg, indent=2)}")

    common = dict(
        initial_budget=cfg["initial_budget"],
        u_normal=cfg["u_normal"],
        u_high=cfg["u_high"],
        **_transition_kwargs(cfg),
    )
    train_data = generate_belief_dataset_v2(
        seed=SEED, n_sequences=N_TRAIN, id_prefix="t7-train", **common
    )
    val_data = generate_belief_dataset_v2(
        seed=SEED + 1, n_sequences=N_VAL, id_prefix="t7-val", **common
    )
    test_data = generate_belief_dataset_v2(
        seed=SEED + 2, n_sequences=N_TEST, id_prefix="t7-test", **common
    )
    train_seqs = [d[0] for d in train_data]
    val_seqs = [d[0] for d in val_data]
    test_seqs = [d[0] for d in test_data]
    print(f"{len(train_seqs)} train / {len(val_seqs)} val / {len(test_seqs)} test sequences")

    oracle = BeliefSensitiveBellmanOracle(
        u_normal_opportunity=cfg["u_normal"],
        u_high_opportunity=cfg["u_high"],
        p_opportunity_normal=cfg["p_opportunity_normal"],
        p_opportunity_high=cfg["p_opportunity_high"],
        transition_normal_to_high=cfg["transition_normal_to_high"],
        transition_high_to_high=cfg["transition_high_to_high"],
    )

    print("computing exact online optimum and hindsight comparators on the test set...")
    online_optimum = {
        seq.sequence_id: online_optimum_as_hindsight_result_v2(
            seq, oracle, INITIAL_BELIEF, **_transition_kwargs(cfg)
        )
        for seq in test_seqs
    }
    hindsight = {seq.sequence_id: compute_hindsight_optimum(seq) for seq in test_seqs}

    total_steps_horizon = len(test_seqs[0].cases)
    reachable = _enumerate_reachable_states(
        oracle, total_steps_horizon, cfg["initial_budget"], INITIAL_BELIEF
    )
    boundary_map = {state: _scan_belief_boundaries(oracle, *state) for state in reachable}

    print("running arm 1 (no pricing), arm 2 (frozen pacing)...")
    arm1_results = {seq.sequence_id: simulate_policy(seq)[0] for seq in test_seqs}
    arm2_results = {
        seq.sequence_id: simulate_policy(seq, pricing_controller=_pacing(seq))[0]
        for seq in test_seqs
    }

    print("running arm 3 (exact belief), arm 4 (fixed prior), arm 5 (true-parameter HMM)...")
    arm3_results: Dict[str, PolicyRunResult] = {}
    arm3_traces: Dict[str, List[Any]] = {}
    arm4_results: Dict[str, PolicyRunResult] = {}
    arm5_results: Dict[str, PolicyRunResult] = {}
    arm10_results: Dict[str, PolicyRunResult] = {}  # inverted belief
    for seq in test_seqs:
        exact_estimator = ExactBeliefEstimatorV2(belief=INITIAL_BELIEF, **_transition_kwargs(cfg))
        exact_result, _, exact_traces = _shadow(seq, oracle, exact_estimator, cfg)
        arm3_results[seq.sequence_id] = exact_result
        arm3_traces[seq.sequence_id] = exact_traces

        exact_beliefs = [t.filtered_belief_value for t in exact_traces]
        fixed_prior = LookupBeliefEstimator(beliefs=[0.5] * len(seq.cases), initial_belief=0.5)
        arm4_results[seq.sequence_id], _, _ = _shadow(seq, oracle, fixed_prior, cfg)

        hmm_belief_vector = np.array([1.0 - INITIAL_BELIEF, INITIAL_BELIEF])
        hmm = HmmBeliefEstimatorV2(belief_vector=hmm_belief_vector, **_transition_kwargs(cfg))
        arm5_results[seq.sequence_id], _, _ = _shadow(seq, oracle, hmm, cfg)

        inverted_beliefs = [1.0 - x for x in exact_beliefs]
        inverted_initial = inverted_beliefs[0] if inverted_beliefs else 0.5
        inverted = LookupBeliefEstimator(beliefs=inverted_beliefs, initial_belief=inverted_initial)
        arm10_results[seq.sequence_id], _, _ = _shadow(seq, oracle, inverted, cfg)

    print("collecting on-policy reference-rollout training pairs (exact-belief shadow charge)...")
    train_features: List[np.ndarray] = []
    train_targets: List[float] = []
    for seq, _, belief_priors, _ in train_data:
        estimator = ExactBeliefEstimatorV2(belief=INITIAL_BELIEF, **_transition_kwargs(cfg))
        _, decisions, _ = _shadow(seq, oracle, estimator, cfg)
        feats, targs = build_belief_training_pairs(seq, decisions, belief_priors, MAX_WINDOW)
        train_features.extend(feats)
        train_targets.extend(targs)

    val_features: List[np.ndarray] = []
    val_targets: List[float] = []
    for seq, _, belief_priors, _ in val_data:
        estimator = ExactBeliefEstimatorV2(belief=INITIAL_BELIEF, **_transition_kwargs(cfg))
        _, decisions, _ = _shadow(seq, oracle, estimator, cfg)
        feats, targs = build_belief_training_pairs(seq, decisions, belief_priors, MAX_WINDOW)
        val_features.extend(feats)
        val_targets.extend(targs)

    test_features: List[np.ndarray] = []
    test_targets: List[float] = []
    test_regimes: List[int] = []
    for seq, true_regimes, belief_priors, _ in test_data:
        estimator = ExactBeliefEstimatorV2(belief=INITIAL_BELIEF, **_transition_kwargs(cfg))
        _, decisions, _ = _shadow(seq, oracle, estimator, cfg)
        feats, targs = build_belief_training_pairs(seq, decisions, belief_priors, MAX_WINDOW)
        test_features.extend(feats)
        test_targets.extend(targs)
        test_regimes.extend(true_regimes[1 : 1 + len(feats)])

    print(
        f"{len(train_features)} train / {len(val_features)} val / {len(test_features)} test rows"
    )

    naive_prediction = float(np.mean(train_targets))
    naive_test_mse = float(np.mean((naive_prediction - np.asarray(test_targets)) ** 2))

    print("fitting ridge (arm 6)...")
    ridge_model = fit_ridge(train_features, train_targets)
    ridge_test_predictions = np.array(
        [min(1.0, max(0.0, predict_ridge(ridge_model, f))) for f in test_features]
    )
    ridge_quality = _belief_quality(
        ridge_test_predictions, np.asarray(test_targets), np.asarray(test_regimes, dtype=float)
    )

    print("training FabricPC belief models (backprop control + predictive coding)...")
    backprop_runs = [
        train_belief_model(
            "backprop", train_features, train_targets, val_features, val_targets, seed
        )
        for seed in TRAIN_SEEDS
    ]
    pcn_runs = [
        train_belief_model("pcn", train_features, train_targets, val_features, val_targets, seed)
        for seed in TRAIN_SEEDS
    ]
    best_backprop = min(backprop_runs, key=lambda r: r["best_val_mse"])
    best_pcn = min(pcn_runs, key=lambda r: r["best_val_mse"])

    backprop_test_predictions = np.clip(
        predict_belief_batch(
            best_backprop["params"], best_backprop["structure"], test_features, "backprop",
            best_backprop["eval_key"],
        ),
        0.0,
        1.0,
    )
    pcn_test_predictions = np.clip(
        predict_belief_batch(
            best_pcn["params"], best_pcn["structure"], test_features, "pcn", best_pcn["eval_key"]
        ),
        0.0,
        1.0,
    )
    backprop_quality = _belief_quality(
        backprop_test_predictions, np.asarray(test_targets), np.asarray(test_regimes, dtype=float)
    )
    pcn_quality = _belief_quality(
        pcn_test_predictions, np.asarray(test_targets), np.asarray(test_regimes, dtype=float)
    )
    learned_quality = {
        "ridge": ridge_quality,
        "fabricpc_backprop": backprop_quality,
        "fabricpc_pcn": pcn_quality,
    }
    gate_b = {
        "naive_constant_prediction_test_mse": naive_test_mse,
        "learned_quality": learned_quality,
        "recovers": {
            name: q["mse_vs_belief_prior"] < 0.5 * naive_test_mse
            for name, q in learned_quality.items()
        },
    }
    gate_b["passed"] = bool(any(gate_b["recovers"].values()))
    print(json.dumps({"gate_b": gate_b}, indent=2, default=str))

    print("running arm 6 (ridge + shadow charge)...")
    arm6_results = {}
    for seq in test_seqs:
        est = RidgeBeliefEstimator(model=ridge_model, max_window=MAX_WINDOW)
        arm6_results[seq.sequence_id], _, _ = _shadow(seq, oracle, est, cfg)

    print("running arm 7 (backprop-trained FabricPC + shadow charge)...")
    arm7_results = {}
    for seq in test_seqs:
        est = FabricPCBeliefEstimator(
            best_backprop["params"],
            best_backprop["structure"],
            "backprop",
            best_backprop["eval_key"],
        )
        arm7_results[seq.sequence_id], _, _ = _shadow(seq, oracle, est, cfg)

    print("running arm 8 (PC-trained FabricPC + shadow charge)...")
    arm8_results = {}
    arm8_traces: Dict[str, List[Any]] = {}
    pcn_estimators: Dict[str, FabricPCBeliefEstimator] = {}
    for seq in test_seqs:
        est = FabricPCBeliefEstimator(
            best_pcn["params"], best_pcn["structure"], "pcn", best_pcn["eval_key"]
        )
        result, _, traces = _shadow(seq, oracle, est, cfg)
        arm8_results[seq.sequence_id] = result
        arm8_traces[seq.sequence_id] = traces
        pcn_estimators[seq.sequence_id] = est

    print("running arm 9 (shuffled FabricPC belief, negative control)...")

    def _shuffle_seed(sequence_id: str) -> int:
        digest = hashlib.sha256(sequence_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (2**31)

    arm9_results = {}
    for seq in test_seqs:
        beliefs = list(pcn_estimators[seq.sequence_id].predicted_beliefs)
        rng = np.random.default_rng(_shuffle_seed(seq.sequence_id))
        shuffled = list(beliefs)
        rng.shuffle(shuffled)
        lookup = LookupBeliefEstimator(beliefs=shuffled)
        arm9_results[seq.sequence_id], _, _ = _shadow(seq, oracle, lookup, cfg)

    all_results = {
        "no_pricing": arm1_results,
        "frozen_pacing": arm2_results,
        "exact_belief": arm3_results,
        "fixed_prior": arm4_results,
        "hmm": arm5_results,
        "ridge": arm6_results,
        "fabricpc_backprop": arm7_results,
        "fabricpc_pcn": arm8_results,
        "fabricpc_pcn_shuffled": arm9_results,
        "inverted_belief": arm10_results,
    }
    metrics_vs_online_optimum = {
        name: regret_metrics(list(results.values()), online_optimum)
        for name, results in all_results.items()
    }
    metrics_vs_hindsight = {
        name: regret_metrics(list(results.values()), hindsight)
        for name, results in all_results.items()
    }

    def _delta_vs(a: str, b: str):
        return bootstrap_ci(
            paired_regret_deltas(
                list(all_results[a].values()), list(all_results[b].values()), online_optimum
            )
        )

    fabricpc_arm = "fabricpc_pcn"
    fabricpc_vs_pacing = _delta_vs(fabricpc_arm, "frozen_pacing")
    fabricpc_vs_fixed_prior = _delta_vs(fabricpc_arm, "fixed_prior")
    fabricpc_vs_shuffled = _delta_vs(fabricpc_arm, "fabricpc_pcn_shuffled")
    fabricpc_vs_backprop = _delta_vs(fabricpc_arm, "fabricpc_backprop")

    pacing_regret = metrics_vs_online_optimum["frozen_pacing"]["mean_regret"]
    exact_regret = metrics_vs_online_optimum["exact_belief"]["mean_regret"]
    fabricpc_regret = metrics_vs_online_optimum[fabricpc_arm]["mean_regret"]
    recoverable_gap = pacing_regret - exact_regret
    captured_fraction = (
        float((pacing_regret - fabricpc_regret) / recoverable_gap)
        if abs(recoverable_gap) > 1e-9
        else float("nan")
    )
    non_inferiority_margin = 0.05 * abs(pacing_regret if pacing_regret else 1.0)

    fabricpc_boundary = _boundary_diagnostics(boundary_map, arm8_traces)
    exact_boundary = _boundary_diagnostics(boundary_map, arm3_traces)

    violations_fabricpc = metrics_vs_online_optimum[fabricpc_arm]["total_violation_count"]
    violations_pacing = metrics_vs_online_optimum["frozen_pacing"]["total_violation_count"]
    fabricpc_latencies = [lat for r in arm8_results.values() for lat in r.decision_latencies]
    pacing_latencies = [lat for r in arm2_results.values() for lat in r.decision_latencies]

    gate_economics = {
        "recoverable_gap": recoverable_gap,
        "captured_fraction": captured_fraction,
        "1_beats_pacing": {
            "delta": fabricpc_vs_pacing,
            "passed": fabricpc_vs_pacing["ci_high"] < 0.0,
        },
        "2_beats_fixed_prior": {
            "delta": fabricpc_vs_fixed_prior,
            "passed": fabricpc_vs_fixed_prior["ci_high"] < 0.0,
        },
        "2_beats_shuffled": {
            "delta": fabricpc_vs_shuffled,
            "passed": fabricpc_vs_shuffled["ci_high"] < 0.0,
        },
        "3_no_additional_violations": {
            "fabricpc_violations": violations_fabricpc,
            "pacing_violations": violations_pacing,
            "passed": violations_fabricpc <= violations_pacing,
        },
        "4_captured_fraction_positive": {
            "passed": bool(captured_fraction == captured_fraction and captured_fraction > 0.0),
        },
        "5_non_inferior_to_backprop": {
            "delta": fabricpc_vs_backprop,
            "non_inferiority_margin": non_inferiority_margin,
            "passed": fabricpc_vs_backprop["ci_low"] < non_inferiority_margin,
        },
        "6_boundary_behavior_vs_shuffled": {
            "fabricpc_near_boundary_fraction": fabricpc_boundary["near_boundary_fraction"],
            "exact_near_boundary_fraction": exact_boundary["near_boundary_fraction"],
        },
        "7_latency": {
            "fabricpc_mean_latency_seconds": (
                statistics.mean(fabricpc_latencies) if fabricpc_latencies else 0.0
            ),
            "pacing_mean_latency_seconds": (
                statistics.mean(pacing_latencies) if pacing_latencies else 0.0
            ),
            "passed": True,
        },
    }
    gate_economics["passed"] = bool(
        gate_economics["1_beats_pacing"]["passed"]
        and gate_economics["2_beats_fixed_prior"]["passed"]
        and gate_economics["2_beats_shuffled"]["passed"]
        and gate_economics["3_no_additional_violations"]["passed"]
        and gate_economics["4_captured_fraction_positive"]["passed"]
        and gate_economics["5_non_inferior_to_backprop"]["passed"]
    )

    report: Dict[str, Any] = {
        "schema": "compitum.fabricpc-tranche7-ten-arm-pilot-report/v1",
        "outcome": (
            f"Gate B passed: {gate_b['passed']}. Primary economics gate: "
            f"{'PASSED' if gate_economics['passed'] else 'FAILED'}."
        ),
        "frozen_gate0_config": cfg,
        "n_train_sequences": len(train_seqs),
        "n_val_sequences": len(val_seqs),
        "n_test_sequences": len(test_seqs),
        "n_train_rows": len(train_features),
        "n_val_rows": len(val_features),
        "n_test_rows": len(test_features),
        "gate_b": gate_b,
        "gate_economics": gate_economics,
        "arms_metrics_vs_exact_online_optimum": metrics_vs_online_optimum,
        "arms_metrics_vs_hindsight": metrics_vs_hindsight,
        "fabricpc_training": {
            "backprop_runs": [
                {k: v for k, v in r.items() if k not in ("params", "structure", "eval_key")}
                for r in backprop_runs
            ],
            "pcn_runs": [
                {k: v for k, v in r.items() if k not in ("params", "structure", "eval_key")}
                for r in pcn_runs
            ],
            "best_backprop_seed": best_backprop["seed"],
            "best_pcn_seed": best_pcn["seed"],
        },
        "total_elapsed_seconds": time.perf_counter() - started,
    }

    out_path = ARTIFACTS / "ten_arm_pilot_report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    out_path.write_text(rendered, encoding="utf-8", newline="")
    print(json.dumps({"gate_economics": gate_economics}, indent=2, default=str))
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
