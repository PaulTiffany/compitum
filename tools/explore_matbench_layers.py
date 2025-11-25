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


def _compute_srmf(df: pd.DataFrame) -> pd.DataFrame:
    tmp = Path(".tmp_layers.csv")
    df.to_csv(tmp, index=False)
    try:
        ad = CSVMatbenchAdapter(path=str(tmp))
        rows: List[Dict[str, float]] = []
        for doc in ad.iter_docs():
            st = map_material_to_srmf(doc)
            rows.append({
                "kappa": float(_curvature_kappa(st)),
                "leak": float(_lyapunov_leak(st)),
                "phase": st.current_phase(),
            })
        srmf = pd.DataFrame(rows)
        return srmf
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def _aurc(y: np.ndarray, scores: np.ndarray, ks: List[int]) -> float:
    order = np.argsort(y)[::-1]
    csum = np.cumsum(y[order])
    ordm = np.argsort(scores)[::-1]
    vals = []
    n = len(y)
    for k in ks:
        k = int(max(1, min(n, int(k))))
        oracle = float(csum[k-1])
        model = float(y[ordm[:k]].sum())
        reg = max(0.0, oracle-model)
        vals.append(0.0 if oracle == 0 else reg/abs(oracle))
    xs = np.asarray([int(k) for k in ks], dtype=float)
    ys = np.asarray(vals, dtype=float)
    if xs[-1] <= 0:
        return float(np.trapz(ys, xs))
    return float(np.trapz(ys, xs)/xs[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Explore emergent layers via SRMF features and regret")
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--objective-col", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["max","min"], default="max")
    ap.add_argument("--kmeans", type=int, default=0, help="If >0, k-means cluster SRMF into K layers")
    ap.add_argument("--quantile-layer-on", type=str, default="band_gap", help="Column to quantile-bin into layers")
    ap.add_argument("--quantiles", type=str, default="0.0,0.5,1.0", help="Quantile edges (e.g., 0.0,0.5,1.0)")
    ap.add_argument("--topk-grid", type=str, default="1,5,10")
    ap.add_argument("--lambda-weight", type=float, default=0.0, help="Score = kappa - lambda*leak")
    ap.add_argument("--out-json", type=Path, default=Path("reports/matbench_layers.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/matbench_layers.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.path)
    if args.objective_col not in df.columns:
        raise SystemExit(f"Missing objective column: {args.objective_col}")
    y = df[args.objective_col].astype(float).to_numpy()
    ks = [int(s) for s in args.topk_grid.split(",") if s.strip()]
    srmf = _compute_srmf(df)
    score = srmf["kappa"].to_numpy() - float(args.lambda_weight) * srmf["leak"].to_numpy()

    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    # Quantile-based layers
    col = args.quantile_layer_on
    if col not in df.columns:
        raise SystemExit(f"Missing layer column: {col}")
    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]
    qs = sorted(set([q for q in qs if 0.0 <= q <= 1.0]))
    edges = np.quantile(df[col].to_numpy(dtype=float), qs)
    bins = np.digitize(df[col].to_numpy(dtype=float), edges[1:-1], right=True)
    labels = [f"Q{b+1}" for b in bins]
    for lab in sorted(set(labels)):
        mask = np.array(labels) == lab
        if mask.sum() == 0:
            continue
        aurc = _aurc(y[mask], score[mask], ks)
        records.append({"layer": f"{col}:{lab}", "size": int(mask.sum()), "AURC": float(aurc)})
    summary["quantile_layers"] = {r["layer"]: {"size": r["size"], "AURC": r["AURC"]} for r in records}

    # Optional k-means on SRMF
    if args.kmeans and args.kmeans > 0:
        try:
            from sklearn.cluster import KMeans  # type: ignore
            X = srmf[["kappa","leak"]].to_numpy()
            km = KMeans(n_clusters=int(args.kmeans), random_state=0)
            labs = km.fit_predict(X)
            for c in range(int(args.kmeans)):
                m = labs == c
                if m.sum() == 0:
                    continue
                aurc = _aurc(y[m], score[m], ks)
                records.append({"layer": f"kmeans:{c}", "size": int(m.sum()), "AURC": float(aurc)})
            summary["kmeans_layers"] = {r["layer"]: {"size": r["size"], "AURC": r["AURC"]} for r in records if str(r["layer"]).startswith("kmeans:")}
        except Exception as e:
            summary["kmeans_error"] = str(e)

    # Phase-based grouping
    for ph in sorted(srmf["phase"].unique()):
        m = srmf["phase"] == ph
        if m.sum() == 0:
            continue
        aurc = _aurc(y[m.to_numpy()], score[m.to_numpy()], ks)
        records.append({"layer": f"phase:{ph}", "size": int(m.sum()), "AURC": float(aurc)})
    summary["phase_layers"] = {r["layer"]: {"size": r["size"], "AURC": r["AURC"]} for r in records if str(r["layer"]).startswith("phase:")}

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out_csv, index=False)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote layers: {args.out_csv} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
