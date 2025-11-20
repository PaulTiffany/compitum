#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from compitum.applications.fusion.eval_offline import evaluate_shot_csv
from compitum.applications.fusion.plasma_monitor import PlasmaMonitor, PlasmaMonitorConfig


def curvature_of_param_curve(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    # Parametric curvature via discrete approximation
    # Assumes xs, ys correspond to increasing p; uses central differences
    n = len(xs)
    kappa = np.zeros(n, dtype=float)
    if n < 3:
        return kappa
    for i in range(1, n-1):
        x1, x2, x3 = xs[i-1], xs[i], xs[i+1]
        y1, y2, y3 = ys[i-1], ys[i], ys[i+1]
        # First derivatives
        dx = (x3 - x1) / 2.0
        dy = (y3 - y1) / 2.0
        # Second derivatives
        ddx = x3 - 2*x2 + x1
        ddy = y3 - 2*y2 + y1
        denom = (dx*dx + dy*dy) ** 1.5
        if denom <= 1e-12:
            kappa[i] = 0.0
        else:
            kappa[i] = abs(dx*ddy - dy*ddx) / denom
    # Endpoints set to neighbor value for convenience
    if n >= 2:
        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]
    return kappa


def evaluate_group(data_dir: Path, p: float, state_dim: int, rank: int,
                   curvature_alarm: float, scales: np.ndarray | None) -> Tuple[float, float, float]:
    cfg = PlasmaMonitorConfig(state_dim=state_dim, rank=rank, curvature_alarm=curvature_alarm,
                              scales=scales, norm_p=p)
    pm = PlasmaMonitor(cfg)
    rows = []
    for csv in sorted(data_dir.glob('*.csv')):
        res = evaluate_shot_csv(csv, monitor=pm, state_dim=state_dim)
        rows.append(res)
    # Risk: miss rate (no alarm before crash)
    miss_list = []
    far_list = []
    lead_list = []
    for r in rows:
        if r.crash_index is not None:
            miss_list.append(0.0 if (r.alarm_index is not None and r.alarm_index < r.crash_index) else 1.0)
            if r.lead_time_ms is not None:
                lead_list.append(float(r.lead_time_ms))
        else:
            # Non-crash shot: count false alarm if any alarm occurred
            far_list.append(1.0 if (r.alarm_index is not None) else 0.0)
    risk = float(np.mean(miss_list)) if miss_list else float('nan')
    far = float(np.mean(far_list)) if far_list else 0.0
    med_lead = float(np.median(lead_list)) if lead_list else float('nan')
    return risk, far, med_lead


def main() -> int:
    ap = argparse.ArgumentParser(description='Lp sweep (Lp Sweep) for fusion offline evaluation')
    ap.add_argument('data_dir', type=Path, help='Directory of shot CSVs (time_ms,q_min[,Te_core,ne,...])')
    ap.add_argument('--state-dim', type=int, default=8)
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--curvature-alarm', type=float, default=0.5)
    ap.add_argument('--p-grid', type=str, default='1.0,1.25,1.5,1.75,2.0', help='Comma-separated p values in [1,2]')
    ap.add_argument('--lambda', dest='lam', type=float, default=1.0, help='Weight for HS in Omega = HR + lam * HS')
    ap.add_argument('--out', type=Path, default=Path('reports/fusion_Lp Sweep_lp.csv'))
    args = ap.parse_args()

    # Optional: load per-dim scales from a file later; for now None
    scales = None

    ps: List[float] = [float(x.strip()) for x in args.p_grid.split(',') if x.strip()]
    rows = []
    for p in ps:
        risk, far, med_lead = evaluate_group(args.data_dir, p, args.state_dim, args.rank, args.curvature_alarm, scales)
        rows.append({'p': p, 'risk': risk, 'far': far, 'median_lead_ms': med_lead})
    df = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)

    # Normalize for L-curve curvature
    x = df['far'].to_numpy(copy=True)
    y = df['risk'].to_numpy(copy=True)
    # Normalize to [0,1] when possible
    if np.isfinite(x).any():
        x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
    if np.isfinite(y).any():
        y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-12)
    kappa = curvature_of_param_curve(x, y)
    df['kappa'] = kappa

    # Omega objective
    df['omega'] = df['risk'] + args.lam * df['far']

    # Picks
    idx_k = int(np.nanargmax(df['kappa'].to_numpy())) if len(df) else 0
    idx_o = int(np.nanargmin(df['omega'].to_numpy())) if len(df) else 0
    p_k = float(df.loc[idx_k, 'p'])
    p_o = float(df.loc[idx_o, 'p'])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote Lp sweep table: {args.out} ({len(df)} rows)")
    print('Top picks:')
    print(f"  curvature argmax p*: {p_k}")
    print(f"  omega argmin p*:    {p_o}  (lambda={args.lam})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

