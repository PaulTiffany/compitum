from pathlib import Path

import numpy as np
import pandas as pd

from compitum.applications.fusion.eval_offline import evaluate_shot_csv
from compitum.applications.fusion.plasma_monitor import PlasmaMonitor, PlasmaMonitorConfig


def test_lead_time_zero_when_alarm_at_crash(tmp_path: Path):
    # q_min below 1.0 from the start -> crash at index 0
    t = np.arange(5, dtype=float)
    q = np.array([0.9, 0.95, 0.97, 0.99, 0.98], dtype=float)
    Te_core = 10.0 - 0.05 * t
    ne = np.full_like(t, 1e20)
    df = pd.DataFrame({"time_ms": t, "Te_core": Te_core, "ne": ne, "q_min": q})
    p = tmp_path / "shot_edge.csv"
    df.to_csv(p, index=False)

    # Extremely low threshold to ensure alarm immediately
    cfg = PlasmaMonitorConfig(state_dim=8, rank=4, curvature_alarm=0.0)
    pm = PlasmaMonitor(cfg)
    res = evaluate_shot_csv(p, monitor=pm, state_dim=8)
    assert res.crash_index == 0
    assert res.alarm_index is not None and res.alarm_index >= res.crash_index
    assert res.lead_time_ms == 0.0


