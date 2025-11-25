#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from compitum.applications.supercon.sc_monitor import SuperconMonitor, SuperconMonitorConfig


def load_dataset(path: Path, state_dim: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [f"x{i+1}" for i in range(state_dim)]
    for c in cols + ["label_sc"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    return df


def evaluate_dir(data_dir: Path, state_dim: int, rank: int, alarm: float) -> pd.DataFrame:
    cfg = SuperconMonitorConfig(state_dim=state_dim, rank=rank, alarm_threshold=alarm)
    mon = SuperconMonitor(cfg)
    rows: List[Dict[str, object]] = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        df = load_dataset(p, state_dim)
        tp = fp = tn = fn = 0
        for _, r in df.iterrows():
            x = r[[f"x{i+1}" for i in range(state_dim)]].to_numpy(dtype=float)
            out = mon.ingest_features(x)
            pred = bool(out["alarm_status"])
            lab = bool(int(r["label_sc"]))
            if pred and lab:
                tp += 1
            elif pred and not lab:
                fp += 1
            elif not pred and not lab:
                tn += 1
            else:
                fn += 1
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        rows.append({"file": p.name, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec, "accuracy": acc})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulated superconductivity offline evaluation")
    ap.add_argument("data_dir", type=Path, help="Directory of CSV files with x1..xD and label_sc")
    ap.add_argument("--state-dim", type=int, default=8)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alarm", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("reports/supercon_offline_metrics.csv"))
    args = ap.parse_args()

    df = evaluate_dir(args.data_dir, args.state_dim, args.rank, args.alarm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote metrics: {args.out} ({len(df)} files)")
    if not df.empty:
        print(df.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
