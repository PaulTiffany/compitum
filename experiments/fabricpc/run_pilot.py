"""Observation-only FabricPC x Compitum pilot (runs under .venv-fabricpc).

Design (see docs/adr/0001-fabricpc-trajectory-observer.md):

- Routing happens ONCE per case with frozen v0.2.0 code; FabricPC evidence is
  collected alongside and never consulted by the router. The experiment arms
  differ only in which features a small held-out predictor may use:

    arm 1 (baseline):            frozen Compitum certificate features only
    arm 2 (terminal-only):       baseline + FabricPC terminal energy
    arm 3 (trajectory-summary):  baseline + trajectory summary features
    arm 4 (negative control):    baseline + arm-3 features computed from
                                 step-shuffled, case-permuted trajectories

- Targets are deferral-need proxies computable from the frozen router itself:
  the top-2 utility margin (small margin = ambiguous route) and the boundary
  gap. The question of record is incremental held-out information BEYOND the
  baseline features, not whether FabricPC predicts anything in isolation.

- One case uses a deliberately non-finite clamp so the governed
  invalid-evidence path is exercised end to end in a real run.

This is a pipeline-validation pilot on synthetic toy routing with N=23
usable cases: it is deliberately underpowered for scientific conclusions and
its report says so. It exists to prove the instruments, artifacts, latency
accounting, and negative-control machinery work before any larger study.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fabricpc_probe import (  # noqa: E402
    DEFAULT_CHECKOUT,
    RECEIPT_PATH,
    observe_case,
    paired_orientation_payload,
    shuffled_control_payload,
)

from compitum.trajectory import (  # noqa: E402
    FabricPCTrajectoryObserver,
    TrajectoryRequest,
    blockwise_audit,
    orientation_audit,
    second_order_audit,
)
from compitum.trajectory.artifacts import write_observation_bundle  # noqa: E402
from compitum.trajectory.evidence import build_evidence  # noqa: E402
from compitum.trajectory.sensors import SECOND_ORDER_INPUT_SCHEMA  # noqa: E402

ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "artifacts"
PROMPT_STYLES = ("simple query", "general query", "complex query")
N_CASES = 24
TRAIN_N = 16


def build_router():
    from compitum.boundary import BoundaryAnalyzer
    from compitum.cli import _load_constraints, _toy_models
    from compitum.coherence import CoherenceFunctional
    from compitum.constraints import ReflectiveConstraintSolver
    from compitum.control import LyapunovController
    from compitum.energy import SymbolicFreeEnergy
    from compitum.metric import SymbolicManifoldMetric
    from compitum.pgd import RegexPromptExtractor
    from compitum.predictors import CalibratedPredictor
    from compitum.router import CompitumRouter

    D = 32
    models = _toy_models(D)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, D))
    predictors = {}
    for m in models:
        entry = {}
        for kind in ("quality", "latency", "cost"):
            p = CalibratedPredictor()
            p.fit(X, rng.random(256))
            entry[kind] = p
        predictors[m.name] = entry
    metrics = {m.name: SymbolicManifoldMetric(D, rank=8, delta=1e-3) for m in models}
    A, B = _load_constraints(REPO_ROOT / "configs" / "constraints_us_default.yaml")
    return CompitumRouter(
        models,
        predictors,
        ReflectiveConstraintSolver(A, B),
        CoherenceFunctional(k=64),
        BoundaryAnalyzer(0.05, 0.65, 0.12),
        LyapunovController(),
        RegexPromptExtractor(),
        metrics,
        SymbolicFreeEnergy(0.5, 0.1, 0.1, 0.0, 0.05),
        update_stride=64,
        enable_metric_update=False,
        enable_controller=True,
    )


def route_cases() -> List[Dict[str, Any]]:
    np.random.seed(20260723)
    router = build_router()
    cases = []
    for i in range(N_CASES):
        prompt = f"{PROMPT_STYLES[i % 3]} {i}"
        emb = np.random.default_rng(1000 + i).standard_normal(32).astype(np.float32)
        started = time.perf_counter()
        certificate = router.route(prompt, embedding=emb)
        route_seconds = time.perf_counter() - started
        cert = json.loads(certificate.to_json())
        # frozen BoundaryAnalyzer reports the top-2 utility margin directly
        margin = cert["boundary"].get("utility_gap")
        margin = float(margin) if margin is not None else None
        cases.append(
            {
                "case_id": f"case-{i:03d}",
                "prompt": prompt,
                "embedding_seed": 1000 + i,
                "certificate": cert,
                "route_seconds": route_seconds,
                "utility_margin": margin,
            }
        )
    return cases


def baseline_features(case: Dict[str, Any]) -> List[float]:
    """Frozen-certificate features. The boundary's ``utility_gap`` is the
    prediction TARGET, so it is deliberately excluded here; ``entropy`` and
    ``uncertainty`` stay in, matching what a deferral policy could see."""
    cert = case["certificate"]
    comps = cert["utility_components"]
    boundary = cert["boundary"]
    drift = cert["drift"]
    return [
        cert["utility"],
        comps["quality"],
        comps["latency"],
        comps["cost"],
        comps["distance"],
        comps["evidence"],
        comps["uncertainty"],
        float(boundary.get("entropy", 0.0)),
        float(boundary.get("uncertainty", 0.0)),
        float(drift.get("drift_ema", 0.0)),
        float(drift.get("trust_radius", 0.0)),
    ]


def trajectory_features(evidence_payload: Dict[str, Any]) -> List[float]:
    convergence = evidence_payload["convergence"]
    energy = evidence_payload["energy_trajectory"]
    first_drop = energy[0] - energy[1] if len(energy) > 1 else 0.0
    per_node = evidence_payload["per_node"]
    return [
        convergence["terminal_total_energy"],
        convergence["energy_reduction_ratio"],
        convergence["monotone_decreasing_fraction"],
        convergence["terminal_latent_grad_norm_total"],
        first_drop,
        per_node["hidden"]["terminal_energy"],
        per_node["latent"]["terminal_energy"],
    ]


def terminal_only_features(evidence_payload: Dict[str, Any]) -> List[float]:
    return [evidence_payload["terminal"]["total_energy"]]


def ridge_holdout_mse(features: List[List[float]], targets: List[float]) -> float:
    """Deterministic ridge regression, train on the first TRAIN_N cases."""
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    X_train, X_test = X[:TRAIN_N], X[TRAIN_N:]
    y_train, y_test = y[:TRAIN_N], y[TRAIN_N:]
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    Xn_train = (X_train - mean) / scale
    Xn_test = (X_test - mean) / scale
    A = Xn_train.T @ Xn_train + 1.0 * np.eye(Xn_train.shape[1])
    w = np.linalg.solve(A, Xn_train.T @ (y_train - y_train.mean()))
    predictions = Xn_test @ w + y_train.mean()
    return float(np.mean((predictions - y_test) ** 2))


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "schema": "compitum.fabricpc-observation-pilot-report/v1",
        "design": "observation-only; FabricPC evidence never consulted by the router",
        "n_cases": N_CASES,
        "train_test_split": [TRAIN_N, N_CASES - 1 - TRAIN_N],
    }

    # ---- routing (frozen v0.2.0 behavior) -------------------------------
    cases = route_cases()
    route_latencies = sorted(c["route_seconds"] for c in cases)

    # ---- observation ----------------------------------------------------
    def runner(req: TrajectoryRequest) -> Dict[str, Any]:
        clamp = req.config["clamp"]
        return observe_case(
            req.case_id,
            (clamp[0], clamp[1]),
            parameter_seed=req.seeds["parameter_seed"],
            state_seed=req.seeds["state_seed"],
        )

    observer = FabricPCTrajectoryObserver(
        runner=runner,
        receipt_path=RECEIPT_PATH,
        checkout=DEFAULT_CHECKOUT,
    )
    observations: List[Dict[str, Any]] = []
    observe_latencies: List[float] = []
    failures = 0
    for index, case in enumerate(cases):
        emb = np.random.default_rng(case["embedding_seed"]).standard_normal(32)
        # case -> observation inputs: clamp from the first two (bounded)
        # embedding coordinates; the LAST case injects a non-finite clamp to
        # exercise the governed invalid path in a real run.
        if index == N_CASES - 1:
            clamp = (float("nan"), 0.0)
        else:
            clamp = (float(np.tanh(emb[0])), float(np.tanh(emb[1])))
        request = TrajectoryRequest(
            case_id=case["case_id"],
            seeds={"parameter_seed": 17, "state_seed": 100 + index},
            config={"clamp": [clamp[0], clamp[1]], "arm": "observation"},
        )
        started = time.perf_counter()
        evidence = observer.observe(request)
        observe_latencies.append(time.perf_counter() - started)
        payload = evidence.to_dict()
        raw = runner(request) if evidence.status == "observed" else None
        bundle = write_observation_bundle(
            ARTIFACTS / "bundles" / case["case_id"],
            evidence,
            raw,
            manifest_extra={"pilot": "observation-only", "case_index": index},
        )
        if evidence.status != "observed":
            failures += 1
        observations.append(
            {
                "case_id": case["case_id"],
                "status": evidence.status,
                "reason": evidence.reason,
                "evidence": payload,
                "bundle_sha256": bundle.bundle_sha256,
            }
        )

    usable = [
        (case, obs)
        for case, obs in zip(cases, observations)
        if obs["status"] == "observed" and case["utility_margin"] is not None
    ]
    report["observed"] = len(usable)
    report["governed_failures"] = {
        "count": failures,
        "detail": [
            {"case_id": o["case_id"], "status": o["status"], "reason": o["reason"]}
            for o in observations
            if o["status"] != "observed"
        ],
    }

    # ---- negative control: shuffled steps + case-permuted pairing -------
    control_features: List[List[float]] = []
    permuted = [usable[(k + 7) % len(usable)][1] for k in range(len(usable))]
    for k, (_case, _obs) in enumerate(usable):
        mismatched = permuted[k]["evidence"]
        raw_like = {
            "schema": "compitum.fabricpc-observation-raw/v1",
            "run_id": "control",
            "node_order": ["source", "hidden", "latent"],
            "steps": mismatched["per_step"],
            "dependency_repository": mismatched["dependency_repository"],
            "dependency_commit": mismatched["dependency_commit"],
            "runtime_seconds": 0.0,
            "config": {},
        }
        shuffled = shuffled_control_payload({**raw_like, "run_id": f"ctl-{k}"}, shuffle_seed=k)
        control_evidence = build_evidence(
            shuffled,
            TrajectoryRequest(case_id=f"control-{k}"),
            "fabricpc-shuffled-control",
            "1",
        )
        control_features.append(trajectory_features(control_evidence.to_dict()))

    # ---- incremental-information comparison -----------------------------
    margins = [case["utility_margin"] for case, _ in usable]
    base = [baseline_features(case) for case, _ in usable]
    arm2 = [b + terminal_only_features(obs["evidence"]) for b, (_, obs) in zip(base, usable)]
    arm3 = [b + trajectory_features(obs["evidence"]) for b, (_, obs) in zip(base, usable)]
    arm4 = [b + ctl for b, ctl in zip(base, control_features)]

    report["incremental_information"] = {
        "target": "top-2 utility margin (deferral-need proxy from the frozen router)",
        "holdout_mse": {
            "arm1_baseline": ridge_holdout_mse(base, margins),
            "arm2_terminal_only": ridge_holdout_mse(arm2, margins),
            "arm3_trajectory_summary": ridge_holdout_mse(arm3, margins),
            "arm4_shuffled_control": ridge_holdout_mse(arm4, margins),
        },
        "reading": (
            "lower MSE than arm1 on held-out cases would indicate incremental "
            "information; the shuffled/mismatched arm4 calibrates how much "
            "apparent improvement is attributable to extra free parameters"
        ),
    }

    # ---- instrument runs against the real pin ---------------------------
    paired = paired_orientation_payload(
        perturbation=1e-3, direction=(1.0, 0.0), parameter_seed=17, state_seed=23
    )
    orientation_certificate = orientation_audit(paired)
    block_certificate = blockwise_audit(
        paired["base_states"],
        paired["probe_states"],
        {name: tuple(rng) for name, rng in paired["blocks"].items()},
    )

    def square(nonlinear: bool) -> Dict[str, Any]:
        eps = 0.1
        first = paired_orientation_payload(eps, (1.0, 0.0), 17, 23, nonlinear=nonlinear)
        second = paired_orientation_payload(eps, (0.0, 1.0), 17, 23, nonlinear=nonlinear)
        combined = paired_orientation_payload(
            math.sqrt(2.0) * eps, (1.0, 1.0), 17, 23, nonlinear=nonlinear
        )
        if first["base_states"] != second["base_states"]:
            raise RuntimeError("square branches disagree on the base trajectory")
        return second_order_audit(
            {
                "schema": SECOND_ORDER_INPUT_SCHEMA,
                "run_id": f"square-nonlinear{int(nonlinear)}",
                "dependency_repository": first["dependency_repository"],
                "dependency_commit": first["dependency_commit"],
                "thresholds": {"residue_norm": 1e-7},
                "base_states": first["base_states"],
                "first_states": first["probe_states"],
                "second_states": second["probe_states"],
                "combined_states": combined["probe_states"],
            }
        )

    linear_square = square(nonlinear=False)
    nonlinear_square = square(nonlinear=True)
    report["instruments"] = {
        "orientation": {
            "candidates": orientation_certificate["candidate_count"],
            "transitions": orientation_certificate["transition_count"],
        },
        "blockwise_first_step": block_certificate["first_step"],
        "second_order": {
            "linear_steps_above_threshold": linear_square["steps_with_second_order_residue"],
            "nonlinear_steps_above_threshold": nonlinear_square["steps_with_second_order_residue"],
            "linear_max_residue": linear_square["max_residue_norm"],
            "nonlinear_max_residue": nonlinear_square["max_residue_norm"],
        },
    }
    for name, payload in (
        ("orientation_certificate.json", orientation_certificate),
        ("blockwise_certificate.json", block_certificate),
        ("second_order_linear_certificate.json", linear_square),
        ("second_order_nonlinear_certificate.json", nonlinear_square),
    ):
        (ARTIFACTS / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="",
        )

    # ---- latency accounting ---------------------------------------------
    def pct(values: List[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    report["latency_seconds"] = {
        "route_p50": statistics.median(route_latencies),
        "route_p95": pct(route_latencies, 0.95),
        "observe_p50": statistics.median(observe_latencies),
        "observe_p95": pct(observe_latencies, 0.95),
        "observe_max": max(observe_latencies),
        "note": (
            "the first observation pays one-off JAX compilation; observation "
            "runs alongside routing and does not gate route selection in this "
            "tranche"
        ),
    }
    report["honest_limitations"] = [
        f"N={len(usable)} usable synthetic cases: far too small for scientific conclusions",
        "toy routing pool and toy FabricPC graph: pipeline validation, not evidence of value",
        "targets are frozen-router proxies (utility margin), not realized outcome labels",
        "any apparent arm-3 gain must beat arm-4 (shuffled control) before it means anything",
    ]
    (ARTIFACTS / "pilot_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline=""
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "observed",
                    "governed_failures",
                    "incremental_information",
                    "instruments",
                    "latency_seconds",
                )
            },
            indent=2,
        )
    )
    print(f"\nreport -> {ARTIFACTS / 'pilot_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
