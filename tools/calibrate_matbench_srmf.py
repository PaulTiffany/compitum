#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from compitum.integrations.matbench_adapter import CSVMatbenchAdapter
from compitum.integrations.materials_project_audit import (
    map_material_to_srmf,
    _curvature_kappa,
    _lyapunov_leak,
)


def _compute_kappa_leak(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    tmp_path = Path(".tmp_matbench_calib_input.csv")
    df.to_csv(tmp_path, index=False)
    try:
        adapter = CSVMatbenchAdapter(path=str(tmp_path))
        kappas: List[float] = []
        leaks: List[float] = []
        for doc in adapter.iter_docs():
            state = map_material_to_srmf(doc)
            kappas.append(_curvature_kappa(state))
            leaks.append(_lyapunov_leak(state))
        return np.asarray(kappas, dtype=float), np.asarray(leaks, dtype=float)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def _topk_regret(y: np.ndarray, scores: np.ndarray, ks: List[int], mode: str) -> List[Dict[str, float]]:
    if mode == "min":
        u = -y
    else:
        u = y
    order_oracle = np.argsort(u)[::-1]
    order_model = np.argsort(scores)[::-1]
    cumsum_oracle = np.cumsum(u[order_oracle])
    out: List[Dict[str, float]] = []
    n = len(y)
    for k in ks:
        k = int(max(1, min(n, k)))
        oracle_sum = float(cumsum_oracle[k - 1])
        model_sum = float(u[order_model[:k]].sum())
        regret = max(0.0, oracle_sum - model_sum)
        norm = 0.0 if oracle_sum == 0.0 else regret / abs(oracle_sum)
        out.append({"k": float(k), "regret_norm": norm})
    return out


def _aurc(rows: List[Dict[str, float]]) -> float:
    if not rows:
        return 0.0
    rs = sorted(rows, key=lambda r: r["k"])  # type: ignore
    xs = np.asarray([r["k"] for r in rs], dtype=float)
    ys = np.asarray([r["regret_norm"] for r in rs], dtype=float)
    if xs[-1] <= 0:
        return float(np.trapz(ys, xs))
    return float(np.trapz(ys, xs) / xs[-1])


def _bootstrap_aurc(y: np.ndarray, scores: np.ndarray, ks: List[int], mode: str, *, n_boot: int, seed: int) -> Dict[str, float]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rows = _topk_regret(y[idx], scores[idx], ks, mode)
        vals.append(_aurc(rows))
    arr = np.asarray(vals)
    return {"lo": float(np.quantile(arr, 0.025)), "hi": float(np.quantile(arr, 0.975))}


def _parse_grid(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_lambdas(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate SRMF lambda for Matbench regret (offline)")
    ap.add_argument("--path", type=Path, required=True, help="CSV with features and objective")
    ap.add_argument("--objective-col", type=str, required=True, help="Objective column name (y_true)")
    ap.add_argument("--mode", type=str, choices=["max", "min"], default="max")
    ap.add_argument("--topk-grid", type=str, default="1,5,10")
    ap.add_argument("--lambda-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=Path("reports/matbench_calibration.json"))
    ap.add_argument("--scores-out", type=Path, default=None, help="Optional CSV with chosen scores (test split)")
    ap.add_argument("--group-col", type=str, default=None, help="Optional per-group lambda calibration (outputs mapping)")
    args = ap.parse_args()

    df = pd.read_csv(args.path)
    for c in ["band_gap", "density", "nsites", "formation_energy_per_atom", args.objective_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")
    n = len(df)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise SystemExit("Invalid split sizes; adjust val/test fractions")
    i_val = idx[n_train : n_train + n_val]
    i_test = idx[n_train + n_val :]

    ks = _parse_grid(args.topk_grid)
    lambdas = _parse_lambdas(args.lambda_grid)

    kappas, leaks = _compute_kappa_leak(df)
    y = df[args.objective_col].astype(float).to_numpy()

    best_lambda = None
    best_val_aurc = float("inf")
    per_lambda: List[Dict[str, Any]] = []
    for lam in lambdas:
        scores_val = kappas[i_val] - float(lam) * leaks[i_val]
        rows_val = _topk_regret(y[i_val], scores_val, ks, args.mode)
        aurc_val = _aurc(rows_val)
        per_lambda.append({"lambda": float(lam), "val_AURC": float(aurc_val)})
        if aurc_val < best_val_aurc:
            best_val_aurc = aurc_val
            best_lambda = float(lam)

    assert best_lambda is not None
    # Evaluate on test with best lambda
    scores_test = kappas[i_test] - best_lambda * leaks[i_test]
    rows_test = _topk_regret(y[i_test], scores_test, ks, args.mode)
    aurc_test = _aurc(rows_test)
    ci = _bootstrap_aurc(y[i_test], scores_test, ks, args.mode, n_boot=args.bootstrap, seed=args.seed)

    out: Dict[str, Any] = {
        "best_lambda": best_lambda,
        "val": {"AURC": best_val_aurc},
        "test": {"AURC": aurc_test, "AURC_CI": ci},
        "grid": per_lambda,
        "topk": ks,
        "splits": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "seed": int(args.seed),
    }

    # Optional per-group calibration
    if args.group_col is not None:
        if args.group_col not in df.columns:
            raise SystemExit(f"Missing group column: {args.group_col}")
        groups = df[args.group_col].astype(str).to_numpy()
        per_group: Dict[str, Any] = {}
        for g in pd.unique(groups):
            mask_val = (groups[i_val] == g)
            mask_test = (groups[i_test] == g)
            if mask_val.sum() == 0 or mask_test.sum() == 0:
                continue
            best_g = None
            best_g_val = float("inf")
            for lam in lambdas:
                sc_val = kappas[i_val][mask_val] - float(lam) * leaks[i_val][mask_val]
                rows_val = _topk_regret(y[i_val][mask_val], sc_val, ks, args.mode)
                aurc_val = _aurc(rows_val)
                if aurc_val < best_g_val:
                    best_g_val = aurc_val
                    best_g = float(lam)
            lamg = 0.0 if best_g is None else float(best_g)
            sc_test = kappas[i_test][mask_test] - lamg * leaks[i_test][mask_test]
            rows_test_g = _topk_regret(y[i_test][mask_test], sc_test, ks, args.mode)
            aurc_test_g = _aurc(rows_test_g)
            ci_g = _bootstrap_aurc(y[i_test][mask_test], sc_test, ks, args.mode, n_boot=args.bootstrap, seed=args.seed)
            per_group[str(g)] = {"lambda": lamg, "test": {"AURC": aurc_test_g, "AURC_CI": ci_g}}
        out["per_group"] = per_group

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote calibration: {args.out_json}")

    if args.scores_out is not None:
        # Write test split scores and y for downstream analysis
        df_out = pd.DataFrame(
            {
                "index": i_test,
                "y_true": y[i_test],
                "kappa": kappas[i_test],
                "leak": leaks[i_test],
                "score": scores_test,
            }
        )
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.scores_out, index=False)
        print(f"Wrote test scores: {args.scores_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
