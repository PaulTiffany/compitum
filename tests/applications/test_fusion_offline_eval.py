from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from compitum.applications.fusion.eval_offline import evaluate_shot_csv
from compitum.applications.fusion.plasma_monitor import PlasmaMonitor, PlasmaMonitorConfig


def _write_sawtooth_csv(tmpdir: Path, steps: int = 60, crash_at: int = 50) -> Path:
    t = np.arange(steps, dtype=float)
    Te_core = 10.0 - 0.05 * t
    ne = np.full_like(t, 1e20)
    q_min = 1.5 - 0.01 * t
    df = pd.DataFrame({"time_ms": t, "Te_core": Te_core, "ne": ne, "q_min": q_min})
    path = tmpdir / "shot.csv"
    df.to_csv(path, index=False)
    return path


def test_lead_time_positive_with_low_alarm(tmp_path: Path):
    csv = _write_sawtooth_csv(tmp_path)
    cfg = PlasmaMonitorConfig(state_dim=8, rank=4, curvature_alarm=1e-6)
    pm = PlasmaMonitor(cfg)
    res = evaluate_shot_csv(csv, monitor=pm, state_dim=8)
    assert res.crash_index is not None
    assert res.alarm_index is not None
    assert res.alarm_index < res.crash_index
    assert res.lead_time_ms is not None and res.lead_time_ms > 0


def test_no_alarm_with_high_threshold(tmp_path: Path):
    csv = _write_sawtooth_csv(tmp_path)
    cfg = PlasmaMonitorConfig(state_dim=8, rank=4, curvature_alarm=1e9)
    pm = PlasmaMonitor(cfg)
    res = evaluate_shot_csv(csv, monitor=pm, state_dim=8)
    assert res.crash_index is not None
    assert res.alarm_index is None
    assert res.lead_time_ms is None

