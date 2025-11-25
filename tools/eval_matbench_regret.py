#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from compitum.integrations.matbench_adapter import CSVMatbenchAdapter
from compitum.integrations.materials_project_audit import (
    map_material_to_srmf,
    _curvature_kappa,
    _lyapunov_leak,
)


def _compute_srmf_components(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (kappa, leak) arrays computed row-wise from df."""
    tmp_path = Path(".tmp_matbench_eval_input.csv")
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


def _compute_srmf_scores(
    df: pd.DataFrame, lambda_weight: float, *, lambda_per_item: np.ndarray | None = None
) -> np.ndarray:
    kappa, leak = _compute_srmf_components(df)
    if lambda_per_item is not None:
        lam = np.asarray(lambda_per_item, dtype=float)
        if lam.shape != kappa.shape:
            raise SystemExit("lambda_per_item shape mismatch")
        return kappa - lam * leak
    return kappa - float(lambda_weight) * leak


def _topk_regret(y: np.ndarray, scores: np.ndarray, ks: List[int], mode: str) -> List[Dict[str, float]]:
    # Define utility u to always maximize
    if mode == "min":
        u = -y
    else:
        u = y
    order_oracle = np.argsort(u)[::-1]
    order_model = np.argsort(scores)[::-1]
    cumsum_oracle = np.cumsum(u[order_oracle])
    # Map from rank -> oracle sum at k
    out: List[Dict[str, float]] = []
    n = len(y)
    for k in ks:
        k = int(k)
        k = max(1, min(n, k))
        oracle_sum = float(cumsum_oracle[k - 1])
        model_idx = order_model[:k]
        model_sum = float(u[model_idx].sum())
        regret = max(0.0, oracle_sum - model_sum)
        norm = 0.0 if oracle_sum == 0.0 else regret / abs(oracle_sum)
        out.append({
            "k": float(k),
            "oracle_sum": oracle_sum,
            "model_sum": model_sum,
            "regret": regret,
            "regret_norm": norm,
        })
    return out


def _aurc(regret_rows: List[Dict[str, float]], key: str = "regret_norm") -> float:
    # Simple trapezoidal area over k in ascending order normalized by k_max
    if not regret_rows:
        return 0.0
    rows = sorted(regret_rows, key=lambda r: r["k"])  # type: ignore
    xs = np.asarray([r["k"] for r in rows], dtype=float)
    ys = np.asarray([r[key] for r in rows], dtype=float)
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
    return {
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
    }


def _parse_grid(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate regret on Matbench-style CSVs (offline-first)")
    ap.add_argument("--path", type=Path, required=True, help="Path to CSV file")
    ap.add_argument("--objective-col", type=str, required=True, help="Response/target column name")
    ap.add_argument("--mode", type=str, choices=["max", "min"], default="max", help="Objective direction")
    ap.add_argument("--score-col", type=str, default=None, help="Optional predicted score column")
    ap.add_argument("--use-srmf", action="store_true", help="If set, derive score from SRMF proxies (kappa - lambda*leak)")
    ap.add_argument("--lambda-weight", type=float, default=0.0, help="Lambda weight for leak in SRMF score")
    ap.add_argument("--topk-grid", type=str, default="1,5,10", help="Comma-separated k values (e.g., 1,5,10)")
    ap.add_argument("--out-csv", type=Path, default=Path("reports/matbench_regret.csv"))
    ap.add_argument("--out-json", type=Path, default=None, help="Optional JSON summary path")
    ap.add_argument("--group-col", type=str, default=None, help="Optional group column for per-group regret")
    ap.add_argument("--out-group-csv", type=Path, default=Path("reports/matbench_regret_groups.csv"))
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap replicates for AURC CI")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap")
    ap.add_argument("--lambda-per-group", type=Path, default=None, help="JSON mapping group->lambda (requires --group-col)")
    ap.add_argument("--cost-col", type=str, default=None, help="Cost column for budget regret")
    ap.add_argument("--budget-grid", type=str, default=None, help="Comma-separated budgets (same units as cost)")
    ap.add_argument("--cost-scale", type=float, default=1.0, help="Scale costs then round to int for knapsack")
    ap.add_argument("--out-budget-csv", type=Path, default=Path("reports/matbench_budget_regret.csv"))
    ap.add_argument("--selection-mode", action="store_true", help="If set, evaluate top-k regret under bootstrap resamples with optional feature noise")
    ap.add_argument("--selection-noise-sigma", type=float, default=0.0, help="Stdev of Gaussian noise to add to features in selection mode")
    args = ap.parse_args()

    df = pd.read_csv(args.path)
    if args.objective_col not in df.columns:
        raise SystemExit(f"Missing objective column: {args.objective_col}")
    y = df[args.objective_col].astype(float).to_numpy()

    lambda_per_item: np.ndarray | None = None
    if args.use_srmf:
        # Ensure required columns
        for c in ["band_gap", "density", "nsites", "formation_energy_per_atom"]:
            if c not in df.columns:
                raise SystemExit(f"--use-srmf requires column '{c}' in CSV")
        if args.lambda_per_group is not None and args.group_col is not None:
            if args.group_col not in df.columns:
                raise SystemExit(f"Missing group column: {args.group_col}")
            import json as _json
            mp = _json.loads(Path(args.lambda_per_group).read_text(encoding="utf-8"))
            groups = df[args.group_col].astype(str).to_numpy()
            lam = np.array([float(mp.get(str(g), args.lambda_weight)) for g in groups], dtype=float)
            lambda_per_item = lam
        scores = _compute_srmf_scores(df, lambda_weight=args.lambda_weight, lambda_per_item=lambda_per_item)
    elif args.score_col is not None:
        if args.score_col not in df.columns:
            raise SystemExit(f"Missing score column: {args.score_col}")
        scores = df[args.score_col].astype(float).to_numpy()
    else:
        raise SystemExit("Provide either --score-col or --use-srmf")

    # Optional feature noise for selection stress
    if args.selection_mode and args.selection_noise_sigma > 0:
        rng = np.random.default_rng(args.seed)
        noise = rng.normal(loc=0.0, scale=args.selection_noise_sigma, size=scores.shape)
        scores = scores + noise

    ks = _parse_grid(args.topk_grid)
    # Selection mode: bootstrap AURC CI via resampling
    if args.selection_mode and args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        n = len(y)
        boots: List[Dict[str, float]] = []
        for b in range(args.bootstrap):
            idx = rng.integers(0, n, size=n)
            rows_b = _topk_regret(y[idx], scores[idx], ks, args.mode)
            boots.append({"AURC": _aurc(rows_b)})
        arr = np.array([b["AURC"] for b in boots], dtype=float)
        rows = _topk_regret(y, scores, ks, args.mode)
        summary: Dict[str, Any] = {
            "AURC": _aurc(rows),
            "AURC_CI": {"lo": float(np.quantile(arr, 0.025)), "hi": float(np.quantile(arr, 0.975))},
            "selection_mode": True,
        }
    else:
        rows = _topk_regret(y, scores, ks, args.mode)
        summary: Dict[str, Any] = {"AURC": _aurc(rows)}

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote regret curves: {args.out_csv}")

    # Per-group analysis (optional)
    if args.group_col is not None:
        if args.group_col not in df.columns:
            raise SystemExit(f"Missing group column: {args.group_col}")
        groups = df[args.group_col].astype(str).to_numpy()
        uniq = pd.unique(groups)
        all_group_rows: List[Dict[str, Any]] = []
        group_summary: Dict[str, Any] = {}
        for g in uniq:
            mask = groups == g
            if mask.sum() == 0:
                continue
            g_rows = _topk_regret(y[mask], scores[mask], ks, args.mode)
            for r in g_rows:
                rr = dict(r)
                rr["group"] = str(g)
                all_group_rows.append(rr)
            g_aurc = _aurc(g_rows)
            entry: Dict[str, Any] = {"AURC": g_aurc}
            if args.bootstrap > 0:
                entry["AURC_CI"] = _bootstrap_aurc(y[mask], scores[mask], ks, args.mode, n_boot=args.bootstrap, seed=args.seed)
            group_summary[str(g)] = entry
        if all_group_rows:
            gdf = pd.DataFrame(all_group_rows)
            args.out_group_csv.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_csv(args.out_group_csv, index=False)
            print(f"Wrote group regret: {args.out_group_csv}")
        summary["groups"] = group_summary
    if args.bootstrap > 0:
        summary["AURC_CI"] = _bootstrap_aurc(y, scores, ks, args.mode, n_boot=args.bootstrap, seed=args.seed)

    # Budget regret (optional)
    if args.cost_col is not None and args.budget_grid is not None:
        if args.cost_col not in df.columns:
            raise SystemExit(f"Missing cost column: {args.cost_col}")
        costs = df[args.cost_col].astype(float).to_numpy()
        budgets = [float(s) for s in args.budget_grid.split(",") if s.strip()]
        util = (-y) if args.mode == "min" else y

        def knapsack_optimal(u: np.ndarray, c: np.ndarray, B: float, scale: float) -> float:
            c_int = np.round(c * scale).astype(int)
            B_int = int(max(0, round(B * scale)))
            n = len(u)
            if B_int <= 0 or n == 0:
                return 0.0
            # Guard against excessive DP size
            if B_int * n > 2_000_000:
                order = np.argsort(u / (c + 1e-12))[::-1]
                total = 0.0
                spent = 0.0
                for i in order:
                    if spent + c[i] <= B + 1e-12:
                        spent += c[i]
                        total += u[i]
                return float(total)
            dp = np.zeros(B_int + 1, dtype=float)
            for i in range(n):
                wi = max(0, c_int[i])
                vi = u[i]
                if wi == 0:
                    dp += vi
                    continue
                for b in range(B_int, wi - 1, -1):
                    dp[b] = max(dp[b], dp[b - wi] + vi)
            return float(dp[B_int])

        def model_under_budget(sc: np.ndarray, u: np.ndarray, c: np.ndarray, B: float) -> float:
            order = np.argsort(sc)[::-1]
            total = 0.0
            spent = 0.0
            for i in order:
                if spent + c[i] <= B + 1e-12:
                    spent += c[i]
                    total += u[i]
            return float(total)

        brow: List[Dict[str, float]] = []
        for B in budgets:
            oracle = knapsack_optimal(util, costs, B, args.cost_scale)
            modelv = model_under_budget(scores, util, costs, B)
            regret = max(0.0, oracle - modelv)
            nrm = 0.0 if oracle == 0.0 else regret / abs(oracle)
            brow.append({"budget": float(B), "oracle": float(oracle), "model": float(modelv), "regret": float(regret), "regret_norm": float(nrm)})
        if brow:
            args.out_budget_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(brow).to_csv(args.out_budget_csv, index=False)
            print(f"Wrote budget regret: {args.out_budget_csv}")
        summary["budget"] = brow
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary: {args.out_json}")
    else:
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
