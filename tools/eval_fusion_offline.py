#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from compitum.applications.fusion.eval_offline import evaluate_dir_csv
from compitum.applications.fusion.plasma_monitor import PlasmaMonitorConfig


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline evaluation for fusion early-warning")
    ap.add_argument(
        "data_dir", type=Path, help="Directory of CSV shots (time_ms,q_min[,Te_core,ne,...])"
    )
    ap.add_argument("--state-dim", type=int, default=8)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--curvature-alarm", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("reports/fusion_offline_metrics.csv"))
    args = ap.parse_args()

    cfg = PlasmaMonitorConfig(
        state_dim=args.state_dim, rank=args.rank, curvature_alarm=args.curvature_alarm
    )
    df = evaluate_dir_csv(args.data_dir, monitor_cfg=cfg, state_dim=args.state_dim)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote metrics: {args.out} ({len(df)} shots)")
    if not df.empty:
        print(df.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
