from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from compitum.applications.fusion.eval_offline import (
    compute_alarm_series,
    evaluate_dir_csv,
    evaluate_shot_csv,
    lead_time_from_q_threshold,
)
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


def test_evaluate_dir_csv_processes_all_shots(tmp_path: Path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    for name, crash_at in (("shot_a", 40), ("shot_b", 55)):
        t = np.arange(60, dtype=float)
        Te_core = 10.0 - 0.05 * t
        ne = np.full_like(t, 1e20)
        q_min = 1.5 - 0.01 * t
        df = pd.DataFrame({"time_ms": t, "Te_core": Te_core, "ne": ne, "q_min": q_min})
        df.to_csv(shots_dir / f"{name}.csv", index=False)

    cfg = PlasmaMonitorConfig(state_dim=8, rank=4, curvature_alarm=1e-6)
    df_out = evaluate_dir_csv(shots_dir, monitor_cfg=cfg, state_dim=8)
    assert list(df_out["shot_id"]) == ["shot_a", "shot_b"]
    assert set(df_out.columns) == {"shot_id", "alarm_index", "crash_index", "lead_time_ms"}
    assert len(df_out) == 2


def test_evaluate_dir_csv_default_config(tmp_path: Path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    _write_sawtooth_csv(shots_dir)
    df_out = evaluate_dir_csv(shots_dir, state_dim=8)
    assert len(df_out) == 1


def test_lead_time_no_crash_ever():
    time_ms = np.arange(10, dtype=float)
    q_min = np.full(10, 5.0)  # never drops below threshold -> no crash
    lead_ms, crash_idx = lead_time_from_q_threshold(q_min, time_ms, alarm_idx=2, q_threshold=1.0)
    assert crash_idx is None
    assert lead_ms is None


def test_lead_time_alarm_at_or_after_crash():
    time_ms = np.arange(10, dtype=float)
    q_min = np.array([5.0] * 3 + [0.5] * 7)  # crash_idx == 3
    lead_ms, crash_idx = lead_time_from_q_threshold(q_min, time_ms, alarm_idx=5, q_threshold=1.0)
    assert crash_idx == 3
    assert lead_ms == 0.0


def test_lead_time_positive_case_exact_value():
    """The one existing "positive lead time" test only checks `> 0` through
    the full PlasmaMonitor pipeline -- never the exact arithmetic
    (`time_ms[crash_idx] - time_ms[alarm_idx]`) in isolation, so a wrong
    operator or swapped indices could still coincidentally stay positive.
    Non-uniform `time_ms` spacing rules out a coincidental match."""
    time_ms = np.array([0.0, 1.0, 3.5, 4.0, 10.0], dtype=float)
    q_min = np.array([5.0, 5.0, 0.5, 0.5, 0.5])  # crash_idx == 2
    lead_ms, crash_idx = lead_time_from_q_threshold(q_min, time_ms, alarm_idx=1, q_threshold=1.0)
    assert crash_idx == 2
    assert lead_ms == 2.5  # time_ms[2] - time_ms[1] == 3.5 - 1.0


class _CannedMonitor:
    """Replays a fixed sequence of `ingest_profile` outputs, independent of
    PlasmaMonitor's real geometry -- isolates `compute_alarm_series`'s own
    latch logic from the metric/controller arithmetic underneath it."""

    def __init__(self, outputs):
        self._outputs = outputs
        self.calls = 0

    def ingest_profile(self, state, t):
        out = self._outputs[self.calls]
        self.calls += 1
        return out


def test_compute_alarm_series_latches_first_alarm_not_last():
    """`if alarm_idx is None and bool(out["alarm_status"])` was only ever
    exercised indirectly through the real PlasmaMonitor, where the alarm
    condition (once triggered by real drift) plausibly never turns back off
    -- never proving the *first-index* latch specifically. A canned monitor
    that alarms at steps 1 and 3, staying quiet in between and after, pins
    both the first-wins semantics and that curvature is read every step."""
    outputs = [
        {"curvature_signal": 0.1, "alarm_status": False},
        {"curvature_signal": 0.9, "alarm_status": True},
        {"curvature_signal": 0.2, "alarm_status": False},
        {"curvature_signal": 0.8, "alarm_status": True},
    ]
    monitor = _CannedMonitor(outputs)
    time_ms = np.arange(4, dtype=float)
    state_seq = np.zeros((4, 1))

    curvature, alarm_idx = compute_alarm_series(monitor, time_ms, state_seq)
    assert alarm_idx == 1
    assert np.allclose(curvature, [0.1, 0.9, 0.2, 0.8])
