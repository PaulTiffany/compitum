#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from compitum.integrations.matbench_adapter import (
    AbstractMatbenchAdapter,
    CSVMatbenchAdapter,
)
from compitum.integrations.materials_project_audit import (
    map_material_to_srmf,
    _curvature_kappa,
    _lyapunov_leak,
)


def _build_adapter(args: argparse.Namespace) -> AbstractMatbenchAdapter:
    if args.adapter != "csv":  # pragma: no cover - future extension
        raise SystemExit("Only --adapter csv is supported in offline mode")
    return CSVMatbenchAdapter(
        path=str(args.path),
        id_column=args.id_col,
        formula_column=args.formula_col,
        label_column=args.label_col,
    )


def _evaluate(
    docs: Iterable[Any],
    *,
    kappa_threshold: float,
    leak_threshold: float,
    label_key: Optional[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        state = map_material_to_srmf(doc)
        kappa = _curvature_kappa(state)
        leak = _lyapunov_leak(state)
        is_cand = (kappa >= float(kappa_threshold)) and (leak <= float(leak_threshold))
        pred = "candidate" if is_cand else "non_candidate"
        row: Dict[str, Any] = dict(
            material_id=getattr(doc, "material_id", ""),
            formula=getattr(doc, "formula_pretty", ""),
            srmf_phase=state.current_phase(),
            curvature_kappa=float(kappa),
            stability_leak=float(leak),
            prediction=pred,
        )
        if label_key is not None and hasattr(doc, label_key):
            val = getattr(doc, label_key)
            try:
                row["label"] = bool(int(val))
            except Exception:
                row["label"] = bool(val)
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    label = df["label"].astype(bool)
    pred_candidate = df["prediction"] == "candidate"
    tp = int((pred_candidate & label).sum())
    fp = int((pred_candidate & (~label)).sum())
    tn = int(((~pred_candidate) & (~label)).sum())
    fn = int(((~pred_candidate) & label).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
    }


def _bootstrap_metrics(
    df: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(df)
    prec: List[float] = []
    rec: List[float] = []
    acc: List[float] = []
    for _ in range(max(0, n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        m = _compute_metrics(df.iloc[idx])
        prec.append(m["precision"])
        rec.append(m["recall"])
        acc.append(m["accuracy"])
    if not prec:
        return {}

    def q(a: List[float], lo: float, hi: float) -> Tuple[float, float]:
        arr = np.asarray(a)
        return float(np.quantile(arr, lo)), float(np.quantile(arr, hi))

    p_lo, p_hi = q(prec, 0.025, 0.975)
    r_lo, r_hi = q(rec, 0.025, 0.975)
    a_lo, a_hi = q(acc, 0.025, 0.975)
    return {
        "precision": {"lo": p_lo, "hi": p_hi},
        "recall": {"lo": r_lo, "hi": r_hi},
        "accuracy": {"lo": a_lo, "hi": a_hi},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate SRMF on Matbench-style CSVs (offline-first)")
    ap.add_argument("--adapter", type=str, default="csv", choices=["csv"], help="Data adapter")
    ap.add_argument("--path", type=Path, required=True, help="Path to CSV file")
    ap.add_argument("--id-col", type=str, default=None, help="Column with material ID")
    ap.add_argument("--formula-col", type=str, default=None, help="Column with pretty formula")
    ap.add_argument(
        "--label-col", type=str, default=None, help="Optional label column (0/1 or bool)"
    )
    ap.add_argument("--kappa-threshold", type=float, default=0.5)
    ap.add_argument("--leak-threshold", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("reports/matbench_srmf.csv"))
    ap.add_argument("--metrics-out", type=Path, default=None, help="Optional JSON for metrics")
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap replicates for CIs")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap")
    args = ap.parse_args()

    adapter = _build_adapter(args)
    df = _evaluate(
        adapter.iter_docs(),
        kappa_threshold=args.kappa_threshold,
        leak_threshold=args.leak_threshold,
        label_key=args.label_col,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote SRMF results: {args.out} ({len(df)} rows)")

    if args.label_col is not None and "label" in df.columns:
        metrics = _compute_metrics(df)
        cis = _bootstrap_metrics(df, n_bootstrap=args.bootstrap, seed=args.seed)
        payload: Dict[str, Any] = {"metrics": metrics, "bootstrap": cis}
        if args.metrics_out is not None:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            with args.metrics_out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote metrics: {args.metrics_out}")
        else:
            print(json.dumps(payload))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
