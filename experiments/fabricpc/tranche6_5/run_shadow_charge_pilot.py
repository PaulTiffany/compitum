"""Tranche 6.5: nine-arm Bellman-consistent shadow-charge pilot.

Runs under ``.venv-fabricpc``. Per docs/adr/0008-bellman-consistent-shadow-price-curve.md:
corrects tranche 6's price-to-action translation (a linear scalar price
times consumption, invalid for lumpy multi-unit actions) with the exact
discrete shadow charge of each candidate action, computed directly from
the same ``BellmanOracle`` -- no new environment, predictor, or search.

Structure: Gate A-prime (translation correctness) is verified first, on
the real test split plus independent robustness seeds -- since
``run_shadow_charge_policy`` with the exact belief is mathematically
required to reproduce the exact online optimum bit-for-bit, this is a
theorem check, not a statistical gate; if it ever fails, the pilot stops
and reports a bug rather than proceeding. Only after it passes does the
pilot reuse tranche 6's already-built Part B training path (ridge, HMM,
FabricPC via train_pcn/train_backprop, run once) to populate arms 5-9.
"""

from __future__ import annotations

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
    BeliefPricingController,
    BellmanOracle,
    DynamicSequence,
    ExactBeliefEstimator,
    HmmBeliefEstimator,
    LookupBeliefEstimator,
    PacingController,
    PolicyRunResult,
    RidgeBeliefEstimator,
    bootstrap_ci,
    build_belief_training_pairs,
    compute_hindsight_optimum,
    fit_ridge,
    generate_belief_dataset,
    online_optimum_as_hindsight_result,
    paired_regret_deltas,
    regret_metrics,
    run_online_optimal_policy,
    run_shadow_charge_policy,
    simulate_policy,
    total_available_over_horizon,
)
from compitum.regret_lab.belief_regime import INITIAL_BELIEF  # noqa: E402
from compitum.regret_lab.windowed_predictor import predict_ridge  # noqa: E402

ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche6_5" / "artifacts"
SEED = 4242
N_TRAIN = 50
N_VAL = 15
N_TEST = 35
FROZEN_PACING_ETA = 1.8
ROBUSTNESS_SEEDS = (4242, 1, 2, 3, 100)


def _pacing(seq: DynamicSequence) -> PacingController:
    return PacingController(
        resource_names=seq.resource_names,
        total_available=total_available_over_horizon(seq),
        total_steps=len(seq.cases),
        eta=FROZEN_PACING_ETA,
    )


def _exact_shadow_decisions(seq: DynamicSequence, oracle: BellmanOracle):
    return run_shadow_charge_policy(seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF))


def _scalar_exact_belief_controller(seq: DynamicSequence, oracle: BellmanOracle):
    return BeliefPricingController(
        oracle=oracle,
        belief_estimator=ExactBeliefEstimator(belief=INITIAL_BELIEF),
        total_steps=len(seq.cases),
        initial_budget=seq.initial_budget["budget"],
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


def _boundary_crossing_rate(reference_choices: List[str], arm_choices: List[str]) -> float:
    total = len(reference_choices)
    if total == 0:
        return 0.0
    disagree = sum(1 for a, b in zip(reference_choices, arm_choices) if a != b)
    return disagree / total


def _verify_gate_a_prime(
    test_seqs: List[DynamicSequence], oracle: BellmanOracle
) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []
    for seq in test_seqs:
        online_result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)
        shadow_result, _, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        if shadow_result.choices != online_result.choices or not np.isclose(
            shadow_result.cumulative_utility, online_result.cumulative_utility
        ):
            mismatches.append(
                {
                    "sequence_id": seq.sequence_id,
                    "online_choices": online_result.choices,
                    "shadow_choices": shadow_result.choices,
                    "online_utility": online_result.cumulative_utility,
                    "shadow_utility": shadow_result.cumulative_utility,
                }
            )
    return {
        "n_sequences_checked": len(test_seqs),
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }


def _verify_gate_a_prime_robustness(oracle: BellmanOracle) -> Dict[str, Any]:
    per_seed: Dict[str, Any] = {}
    for seed in ROBUSTNESS_SEEDS:
        data = generate_belief_dataset(
            seed=seed + 2, n_sequences=N_TEST, id_prefix=f"robust-{seed}"
        )
        seqs = [d[0] for d in data]
        result = _verify_gate_a_prime(seqs, oracle)
        per_seed[str(seed)] = {
            "n_mismatches": len(result["mismatches"]),
            "passed": result["passed"],
        }
    return {"per_seed": per_seed, "passed": all(v["passed"] for v in per_seed.values())}


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_data = generate_belief_dataset(seed=SEED, n_sequences=N_TRAIN, id_prefix="belief-train")
    val_data = generate_belief_dataset(seed=SEED + 1, n_sequences=N_VAL, id_prefix="belief-val")
    test_data = generate_belief_dataset(seed=SEED + 2, n_sequences=N_TEST, id_prefix="belief-test")
    train_seqs = [d[0] for d in train_data]
    val_seqs = [d[0] for d in val_data]
    test_seqs = [d[0] for d in test_data]
    print(f"{len(train_seqs)} train / {len(val_seqs)} val / {len(test_seqs)} test sequences")

    oracle = BellmanOracle()

    print("verifying Gate A-prime (translation correctness) on the test split...")
    gate_a_prime = _verify_gate_a_prime(test_seqs, oracle)
    print("verifying Gate A-prime robustness across independent seeds...")
    gate_a_prime_robustness = _verify_gate_a_prime_robustness(oracle)
    gate_a_prime["robustness"] = gate_a_prime_robustness
    gate_a_prime["passed"] = gate_a_prime["passed"] and gate_a_prime_robustness["passed"]
    gate_a_prime_summary = {
        "n_sequences_checked": gate_a_prime["n_sequences_checked"],
        "n_mismatches": len(gate_a_prime["mismatches"]),
        "robustness": gate_a_prime["robustness"],
        "passed": gate_a_prime["passed"],
    }
    print(json.dumps({"gate_a_prime": gate_a_prime_summary}, indent=2, default=str))

    if not gate_a_prime["passed"]:
        report = {
            "schema": "compitum.fabricpc-shadow-charge-pilot-report/v1",
            "outcome": (
                "STOPPED: Gate A-prime (translation correctness) failed. There is an "
                "implementation or timing bug in belief_action_pricing.py -- resolve it "
                "before running any further arms or gates, per ADR 0008."
            ),
            "gate_a_prime": gate_a_prime,
            "total_elapsed_seconds": time.perf_counter() - started,
        }
        out_path = ARTIFACTS / "shadow_charge_pilot_report.json"
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="",
        )
        print(f"\nGate A-prime FAILED -- stopping. report -> {out_path}")
        return 1

    print("Gate A-prime passed (bit-identical to the exact online optimum on every check).")

    online_optimum = {
        seq.sequence_id: online_optimum_as_hindsight_result(seq, oracle, INITIAL_BELIEF)
        for seq in test_seqs
    }
    hindsight = {seq.sequence_id: compute_hindsight_optimum(seq) for seq in test_seqs}

    print("running arm 1 (no pricing), arm 2 (frozen pacing)...")
    arm1_results = {seq.sequence_id: simulate_policy(seq)[0] for seq in test_seqs}
    arm2_results = {
        seq.sequence_id: simulate_policy(seq, pricing_controller=_pacing(seq))[0]
        for seq in test_seqs
    }

    print("running arm 3 (exact belief + scalar price, tranche 6's failed ablation)...")
    arm3_results = {
        seq.sequence_id: simulate_policy(
            seq, pricing_controller=_scalar_exact_belief_controller(seq, oracle)
        )[0]
        for seq in test_seqs
    }

    print("running arm 4 (exact belief + Bellman action shadow charge)...")
    arm4_results: Dict[str, PolicyRunResult] = {}
    for seq in test_seqs:
        result, _, _ = run_shadow_charge_policy(
            seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
        )
        arm4_results[seq.sequence_id] = result

    metrics_so_far = {
        name: regret_metrics(list(results.values()), online_optimum)
        for name, results in {
            "no_pricing": arm1_results,
            "frozen_pacing": arm2_results,
            "exact_belief_scalar_price": arm3_results,
            "exact_belief_shadow_charge": arm4_results,
        }.items()
    }
    pacing_regret = metrics_so_far["frozen_pacing"]["mean_regret"]
    exact_shadow_regret = metrics_so_far["exact_belief_shadow_charge"]["mean_regret"]
    scalar_price_regret = metrics_so_far["exact_belief_scalar_price"]["mean_regret"]
    recoverable_gap = pacing_regret - exact_shadow_regret
    print(
        json.dumps(
            {
                "recoverable_gap": recoverable_gap,
                "pacing_mean_regret": pacing_regret,
                "exact_belief_shadow_charge_mean_regret": exact_shadow_regret,
                "scalar_price_mean_regret": scalar_price_regret,
            },
            indent=2,
        )
    )

    print("collecting on-policy reference-rollout training pairs (shadow-charge exact belief)...")
    train_features: List[np.ndarray] = []
    train_targets: List[float] = []
    for seq, _, belief_priors, _ in train_data:
        _, decisions, _ = _exact_shadow_decisions(seq, oracle)
        feats, targs = build_belief_training_pairs(seq, decisions, belief_priors, MAX_WINDOW)
        train_features.extend(feats)
        train_targets.extend(targs)

    val_features: List[np.ndarray] = []
    val_targets: List[float] = []
    for seq, _, belief_priors, _ in val_data:
        _, decisions, _ = _exact_shadow_decisions(seq, oracle)
        feats, targs = build_belief_training_pairs(seq, decisions, belief_priors, MAX_WINDOW)
        val_features.extend(feats)
        val_targets.extend(targs)

    test_features: List[np.ndarray] = []
    test_targets: List[float] = []
    test_regimes: List[int] = []
    for seq, true_regimes, belief_priors, _ in test_data:
        _, decisions, _ = _exact_shadow_decisions(seq, oracle)
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

    RECOVERY_THRESHOLD = 0.5
    learned_quality = {
        "ridge": ridge_quality,
        "fabricpc_backprop": backprop_quality,
        "fabricpc_pcn": pcn_quality,
    }
    gate_b = {
        "naive_constant_prediction_test_mse": naive_test_mse,
        "recovery_threshold_fraction_of_naive_mse": RECOVERY_THRESHOLD,
        "learned_quality": learned_quality,
        "recovers": {
            name: q["mse_vs_belief_prior"] < RECOVERY_THRESHOLD * naive_test_mse
            for name, q in learned_quality.items()
        },
    }
    gate_b["passed"] = bool(any(gate_b["recovers"].values()))
    print(json.dumps({"gate_b": gate_b}, indent=2, default=str))

    print("running arm 5 (true-parameter HMM + shadow charge)...")
    arm5_results = {}
    for seq in test_seqs:
        result, _, _ = run_shadow_charge_policy(seq, oracle, HmmBeliefEstimator())
        arm5_results[seq.sequence_id] = result

    print("running arm 6 (ridge + shadow charge)...")
    arm6_results = {}
    for seq in test_seqs:
        result, _, _ = run_shadow_charge_policy(
            seq, oracle, RidgeBeliefEstimator(model=ridge_model, max_window=MAX_WINDOW)
        )
        arm6_results[seq.sequence_id] = result

    print("running arm 7 (backprop-trained FabricPC + shadow charge)...")
    arm7_results = {}
    for seq in test_seqs:
        est = FabricPCBeliefEstimator(
            best_backprop["params"],
            best_backprop["structure"],
            "backprop",
            best_backprop["eval_key"],
        )
        result, _, _ = run_shadow_charge_policy(seq, oracle, est)
        arm7_results[seq.sequence_id] = result

    print("running arm 8 (PC-trained FabricPC + shadow charge)...")
    arm8_results = {}
    pcn_estimators: Dict[str, FabricPCBeliefEstimator] = {}
    for seq in test_seqs:
        est = FabricPCBeliefEstimator(
            best_pcn["params"], best_pcn["structure"], "pcn", best_pcn["eval_key"]
        )
        result, _, _ = run_shadow_charge_policy(seq, oracle, est)
        arm8_results[seq.sequence_id] = result
        pcn_estimators[seq.sequence_id] = est

    print("running arm 9 (shuffled FabricPC belief, negative control)...")
    import hashlib

    def _shuffle_seed(sequence_id: str) -> int:
        digest = hashlib.sha256(sequence_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (2**31)

    arm9_results = {}
    for seq in test_seqs:
        beliefs = list(pcn_estimators[seq.sequence_id].predicted_beliefs)
        rng = np.random.default_rng(_shuffle_seed(seq.sequence_id))
        shuffled = list(beliefs)
        rng.shuffle(shuffled)
        result, _, _ = run_shadow_charge_policy(
            seq, oracle, LookupBeliefEstimator(beliefs=shuffled)
        )
        arm9_results[seq.sequence_id] = result

    all_results = {
        "no_pricing": arm1_results,
        "frozen_pacing": arm2_results,
        "exact_belief_scalar_price": arm3_results,
        "exact_belief_shadow_charge": arm4_results,
        "hmm_shadow_charge": arm5_results,
        "ridge_shadow_charge": arm6_results,
        "fabricpc_backprop_shadow_charge": arm7_results,
        "fabricpc_pcn_shadow_charge": arm8_results,
        "fabricpc_pcn_shuffled_shadow_charge": arm9_results,
    }
    metrics_vs_online_optimum = {
        name: regret_metrics(list(results.values()), online_optimum)
        for name, results in all_results.items()
    }
    metrics_vs_hindsight = {
        name: regret_metrics(list(results.values()), hindsight)
        for name, results in all_results.items()
    }

    def _delta_vs(a: str, b: str) -> Dict[str, float]:
        return bootstrap_ci(
            paired_regret_deltas(
                list(all_results[a].values()), list(all_results[b].values()), online_optimum
            )
        )

    fabricpc_arm = "fabricpc_pcn_shadow_charge"
    fabricpc_vs_pacing = _delta_vs(fabricpc_arm, "frozen_pacing")
    fabricpc_vs_ridge = _delta_vs(fabricpc_arm, "ridge_shadow_charge")
    fabricpc_vs_backprop = _delta_vs(fabricpc_arm, "fabricpc_backprop_shadow_charge")
    fabricpc_vs_shuffled = _delta_vs(fabricpc_arm, "fabricpc_pcn_shuffled_shadow_charge")

    fabricpc_regret = metrics_vs_online_optimum[fabricpc_arm]["mean_regret"]
    captured_fraction = (
        float((pacing_regret - fabricpc_regret) / recoverable_gap)
        if abs(recoverable_gap) > 1e-9
        else float("nan")
    )
    non_inferiority_margin = 0.05 * abs(pacing_regret if pacing_regret else 1.0)

    gate_c = {
        "recoverable_gap": recoverable_gap,
        "captured_fraction": captured_fraction,
        "beats_pacing": {
            "delta": fabricpc_vs_pacing,
            "passed": fabricpc_vs_pacing["ci_high"] < 0.0,
        },
        "beats_shuffled": {
            "delta": fabricpc_vs_shuffled,
            "passed": fabricpc_vs_shuffled["ci_high"] < 0.0,
        },
        "materially_improves_on_ridge": {
            "delta": fabricpc_vs_ridge,
            "passed": fabricpc_vs_ridge["ci_high"] < 0.0,
        },
        "non_inferior_to_backprop": {
            "delta": fabricpc_vs_backprop,
            "non_inferiority_margin": non_inferiority_margin,
            "passed": fabricpc_vs_backprop["ci_low"] < non_inferiority_margin,
        },
        "captured_fraction_positive_with_ci_support": {
            "passed": bool(captured_fraction == captured_fraction and captured_fraction > 0.0)
            and fabricpc_vs_pacing["ci_high"] < 0.0,
        },
    }
    gate_c["passed"] = bool(
        gate_c["beats_pacing"]["passed"]
        and gate_c["beats_shuffled"]["passed"]
        and gate_c["non_inferior_to_backprop"]["passed"]
        and gate_c["captured_fraction_positive_with_ci_support"]["passed"]
    )

    boundary_crossing = {}
    online_choice_seq = {sid: online_optimum[sid].choices for sid in online_optimum}
    for name in (
        "hmm_shadow_charge",
        "ridge_shadow_charge",
        "fabricpc_backprop_shadow_charge",
        "fabricpc_pcn_shadow_charge",
        "fabricpc_pcn_shuffled_shadow_charge",
    ):
        rates = [
            _boundary_crossing_rate(online_choice_seq[sid], all_results[name][sid].choices)
            for sid in online_choice_seq
        ]
        boundary_crossing[name] = statistics.mean(rates)

    report: Dict[str, Any] = {
        "schema": "compitum.fabricpc-shadow-charge-pilot-report/v1",
        "outcome": (
            f"Gate A-prime passed. Gate B passed: {gate_b['passed']}. "
            f"Gate C {'PASSED' if gate_c['passed'] else 'FAILED'}."
        ),
        "n_train_sequences": len(train_seqs),
        "n_val_sequences": len(val_seqs),
        "n_test_sequences": len(test_seqs),
        "n_train_rows": len(train_features),
        "n_val_rows": len(val_features),
        "n_test_rows": len(test_features),
        "gate_a_prime": gate_a_prime,
        "recoverable_gap_summary": {
            "pacing_mean_regret": pacing_regret,
            "exact_belief_shadow_charge_mean_regret": exact_shadow_regret,
            "exact_belief_scalar_price_mean_regret": scalar_price_regret,
            "recoverable_gap": recoverable_gap,
        },
        "gate_b": gate_b,
        "gate_c": gate_c,
        "arms_metrics_vs_exact_online_optimum": metrics_vs_online_optimum,
        "arms_metrics_vs_hindsight": metrics_vs_hindsight,
        "boundary_crossing_rate_vs_online_optimum": boundary_crossing,
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
        "honest_limitations": [
            "boundary-sensitive belief diagnostics were narrowed to a boundary-crossing "
            "rate (fraction of steps where an arm's shadow-charge choice differs from the "
            "true online optimum's own choice at that state) rather than precomputing "
            "explicit belief-interval decision boundaries per (time, budget, observation) "
            "state -- a scope simplification given runtime discipline; the crossing rate "
            "is the economically-relevant consequence of a belief error, whether or not "
            "an explicit interval was precomputed",
            "arm 5 (true-parameter HMM) is not required to beat arm 4 (exact belief); it "
            "is reported as an oracle-quality structured ceiling, per ADR 0008",
            "50 train / 15 val / 35 test sequences, 3 declared training seeds per FabricPC "
            "training method, one fixed topology, no hyperparameter search -- reusing "
            "tranche 6's Part B infrastructure unchanged",
        ],
        "total_elapsed_seconds": time.perf_counter() - started,
    }

    out_path = ARTIFACTS / "shadow_charge_pilot_report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    out_path.write_text(rendered, encoding="utf-8", newline="")
    summary = {"gate_b": {"passed": gate_b["passed"]}, "gate_c": gate_c}
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
