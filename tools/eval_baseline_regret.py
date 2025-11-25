#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _kfold_oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    oof = np.zeros(n, dtype=float)

    if model.lower() == "lgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as e:
            raise SystemExit("LightGBM not installed; use --model ridge or install lightgbm") from e
        for fold in folds:
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            train = ~np.isnan(X[mask]).any(axis=1)
            valid = ~np.isnan(X[~mask]).any(axis=1)
            trX, trY = X[mask][train], y[mask][train]
            vaX = X[~mask][valid]
            if len(trX) == 0 or len(vaX) == 0:
                continue
            model_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=seed)
            model_lgb.fit(trX, trY)
            preds = model_lgb.predict(vaX)
            oof[np.where(~mask)[0][valid]] = preds
    else:
        # Ridge as a stable default without extra deps
        from sklearn.linear_model import Ridge  # type: ignore

        for fold in folds:
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            trX, trY = X[mask], y[mask]
            vaX = X[~mask]
            if len(trX) == 0 or len(vaX) == 0:
                continue
            model_rg = Ridge(alpha=1.0, random_state=seed)
            model_rg.fit(trX, trY)
            preds = model_rg.predict(vaX)
            oof[np.where(~mask)[0]] = preds
    return oof


def _topk_regret(y: np.ndarray, scores: np.ndarray, ks: List[int]) -> List[Dict[str, float]]:
    order_oracle = np.argsort(y)[::-1]
    cumsum_oracle = np.cumsum(y[order_oracle])
    order_model = np.argsort(scores)[::-1]
    out: List[Dict[str, float]] = []
    n = len(y)
    for k in ks:
        k = int(max(1, min(n, int(k))))
        oracle = float(cumsum_oracle[k - 1])
        model = float(y[order_model[:k]].sum())
        reg = max(0.0, oracle - model)
        out.append({
            "k": float(k),
            "regret": reg,
            "regret_norm": 0.0 if oracle == 0.0 else reg / abs(oracle),
        })
    return out


def _aurc(rows: List[Dict[str, float]]) -> float:
    if not rows:
        return 0.0
    xs = np.asarray([r["k"] for r in sorted(rows, key=lambda r: r["k"])], dtype=float)
    ys = np.asarray([r["regret_norm"] for r in sorted(rows, key=lambda r: r["k"])], dtype=float)
    if xs[-1] <= 0:
        return float(np.trapz(ys, xs))
    return float(np.trapz(ys, xs) / xs[-1])


def _bootstrap_aurc(y: np.ndarray, scores: np.ndarray, ks: List[int], *, n_boot: int, seed: int) -> Dict[str, float]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rows = _topk_regret(y[idx], scores[idx], ks)
        vals.append(_aurc(rows))
    arr = np.asarray(vals)
    return {"lo": float(np.quantile(arr, 0.025)), "hi": float(np.quantile(arr, 0.975))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline regret via CV scores (ridge or lgbm)")
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--objective-col", type=str, required=True)
    ap.add_argument("--feature-cols", type=str, default="band_gap,density,nsites,formation_energy_per_atom")
    ap.add_argument("--model", type=str, default="ridge", choices=["ridge", "lgbm"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--topk-grid", type=str, default="1,5,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out-csv", type=Path, default=Path("reports/matbench_baseline_regret.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/matbench_baseline_regret.json"))
    ap.add_argument("--plot", action="store_true", help="If matplotlib is available, emit plots to reports/")
    args = ap.parse_args()

    df = pd.read_csv(args.path)
    if args.objective_col not in df.columns:
        raise SystemExit(f"Missing objective column: {args.objective_col}")
    feat_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    for c in feat_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing feature column: {c}")
    X = df[feat_cols].astype(float).to_numpy()
    y = df[args.objective_col].astype(float).to_numpy()
    ks = [int(s) for s in args.topk_grid.split(",") if s.strip()]

    scores = _kfold_oof_scores(X, y, model=args.model, n_splits=max(2, args.folds), seed=args.seed)
    rows = _topk_regret(y, scores, ks)
    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    summary: Dict[str, Any] = {"AURC": _aurc(rows)}
    if args.bootstrap > 0:
        summary["AURC_CI"] = _bootstrap_aurc(y, scores, ks, n_boot=args.bootstrap, seed=args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote baseline regret: {args.out_csv} and {args.out_json}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            # Regret@k curve
            plt.figure()
            plt.plot(out_df["k"], out_df["regret_norm"], marker="o")
            plt.xlabel("k")
            plt.ylabel("Normalized Regret")
            plt.title("Baseline Regret@k")
            p1 = Path("reports/matbench_baseline_regret_curve.png")
            plt.savefig(p1, dpi=150, bbox_inches="tight")
            plt.close()

            # AURC bar with CI
            plt.figure()
            aurc = summary.get("AURC", 0.0)
            lo = summary.get("AURC_CI", {}).get("lo", aurc)
            hi = summary.get("AURC_CI", {}).get("hi", aurc)
            plt.bar(["AURC"], [aurc], yerr=[[max(0, aurc - lo)], [max(0, hi - aurc)]], capsize=6)
            plt.ylabel("AURC (lower is better)")
            p2 = Path("reports/matbench_baseline_aurc.png")
            plt.savefig(p2, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Wrote plots: {p1}, {p2}")
        except Exception as e:
            print(f"Plotting skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
