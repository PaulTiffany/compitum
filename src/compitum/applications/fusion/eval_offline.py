from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .diiid_adapter import ShotData, load_shot_csv
from .plasma_monitor import PlasmaMonitor, PlasmaMonitorConfig


@dataclass
class LeadTimeResult:
    shot_id: str
    alarm_index: Optional[int]
    crash_index: Optional[int]
    lead_time_ms: Optional[float]


def compute_alarm_series(
    monitor: PlasmaMonitor, time_ms: np.ndarray, state_seq: np.ndarray
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Runs PlasmaMonitor over a sequence and returns curvature over time and the first alarm index.
    """
    curvature = np.zeros(len(time_ms), dtype=float)
    alarm_idx: Optional[int] = None
    for i, (t, x) in enumerate(zip(time_ms, state_seq)):
        out = monitor.ingest_profile(x, float(t))
        curvature[i] = float(out["curvature_signal"])
        if alarm_idx is None and bool(out["alarm_status"]):
            alarm_idx = i
    return curvature, alarm_idx


def lead_time_from_q_threshold(
    q_min: np.ndarray, time_ms: np.ndarray, alarm_idx: Optional[int], q_threshold: float = 1.0
) -> Tuple[Optional[float], Optional[int]]:
    """
    Computes lead time (ms) between the first alarm and the first time q_min < q_threshold.
    Returns (lead_time_ms, crash_idx). If either is missing, returns (None, crash_idx or None).
    """
    crash_idx: Optional[int]
    below = np.nonzero(q_min < float(q_threshold))[0]
    crash_idx = int(below[0]) if len(below) else None
    if alarm_idx is None or crash_idx is None:
        return None, crash_idx
    if alarm_idx >= crash_idx:
        return 0.0, crash_idx
    return float(time_ms[crash_idx] - time_ms[alarm_idx]), crash_idx


def evaluate_shot_csv(path: Path, monitor: PlasmaMonitor, state_dim: int = 8) -> LeadTimeResult:
    data: ShotData = load_shot_csv(path, state_dim=state_dim)
    # Reset equilibrium for each shot
    monitor.reset_equilibrium(data.state[0])
    curv, alarm_idx = compute_alarm_series(monitor, data.time_ms, data.state)
    lead_ms, crash_idx = lead_time_from_q_threshold(
        data.q_min, data.time_ms, alarm_idx, q_threshold=monitor.q_threshold
    )
    return LeadTimeResult(
        shot_id=path.stem, alarm_index=alarm_idx, crash_index=crash_idx, lead_time_ms=lead_ms
    )


def evaluate_dir_csv(
    dir_path: Path, monitor_cfg: Optional[PlasmaMonitorConfig] = None, state_dim: int = 8
) -> pd.DataFrame:
    cfg = monitor_cfg or PlasmaMonitorConfig(state_dim=state_dim)
    monitor = PlasmaMonitor(cfg)

    rows: List[Dict[str, object]] = []
    for p in sorted(Path(dir_path).glob("*.csv")):
        res = evaluate_shot_csv(p, monitor=monitor, state_dim=state_dim)
        rows.append(
            {
                "shot_id": res.shot_id,
                "alarm_index": res.alarm_index,
                "crash_index": res.crash_index,
                "lead_time_ms": res.lead_time_ms,
            }
        )
    return pd.DataFrame(rows)
