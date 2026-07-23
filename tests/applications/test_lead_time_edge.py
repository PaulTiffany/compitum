from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from compitum.applications.fusion.eval_offline import (
    evaluate_dir_csv,
    evaluate_shot_csv,
    lead_time_from_q_threshold,
)
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


def test_lead_time_from_q_threshold_default_is_one():
    """`q_threshold: float = 1.0` was never exercised via its own default --
    evaluate_shot_csv always passes an explicit `q_threshold=monitor.q_threshold`,
    so a mutated default (e.g. 2.0) would never surface through that call
    path. Call the function directly, without q_threshold, to pin the
    default itself: crossing below 1.0 (not 2.0) determines crash_idx."""
    q_min = np.array([2.0, 1.5, 0.5])
    time_ms = np.array([0.0, 10.0, 20.0])
    lead_ms, crash_idx = lead_time_from_q_threshold(q_min, time_ms, alarm_idx=0)
    assert crash_idx == 2  # first index where q_min < 1.0 (default), not < 2.0
    assert lead_ms == 20.0


def test_lead_time_from_q_threshold_strict_less_than_at_exact_equality():
    """`q_min < q_threshold` was mutated to `<=` -- at an exact equality
    (q_min == q_threshold), the real code must NOT count it as a crash
    (strict less-than), while `<=` would. No existing test constructs an
    exact float equality between q_min and q_threshold."""
    q_min = np.array([2.0, 1.0, 2.0])  # exactly equals threshold at index 1
    time_ms = np.array([0.0, 10.0, 20.0])
    lead_ms, crash_idx = lead_time_from_q_threshold(q_min, time_ms, alarm_idx=0, q_threshold=1.0)
    assert crash_idx is None  # 1.0 is not strictly less than 1.0
    assert lead_ms is None


def test_evaluate_shot_csv_default_state_dim_is_eight(tmp_path: Path):
    """`state_dim: int = 8` was never exercised via its own default -- the
    only existing test always passes state_dim=8 explicitly, matching the
    monitor's own (also-explicit) state_dim. Build a monitor with ITS OWN
    default state_dim and call evaluate_shot_csv without overriding
    state_dim either, so a mutated default (e.g. 9) causes a genuine
    dimension mismatch between the loaded state and what the monitor
    expects -- observable as a crash, not a silent wrong value."""
    t = np.arange(5, dtype=float)
    q = np.array([0.9, 0.95, 0.97, 0.99, 0.98], dtype=float)
    Te_core = 10.0 - 0.05 * t
    ne = np.full_like(t, 1e20)
    df = pd.DataFrame({"time_ms": t, "Te_core": Te_core, "ne": ne, "q_min": q})
    p = tmp_path / "shot.csv"
    df.to_csv(p, index=False)

    pm = PlasmaMonitor(PlasmaMonitorConfig())  # default state_dim
    res = evaluate_shot_csv(p, monitor=pm)  # no state_dim kwarg -- must match the monitor's default
    assert res.shot_id == "shot"


def test_evaluate_shot_csv_resets_equilibrium_with_first_state_row(tmp_path: Path):
    """`monitor.reset_equilibrium(data.state[0])` was mutated to `state[1]`
    -- no existing test distinguishes which row gets used, since the
    monitor's own downstream behavior doesn't expose which equilibrium it
    was reset to. Wrap reset_equilibrium to record its argument directly."""
    t = np.arange(5, dtype=float)
    q = np.array([0.9, 0.95, 0.97, 0.99, 0.98], dtype=float)
    Te_core = 10.0 - 0.05 * t
    ne = np.full_like(t, 1e20)
    df = pd.DataFrame({"time_ms": t, "Te_core": Te_core, "ne": ne, "q_min": q})
    p = tmp_path / "shot.csv"
    df.to_csv(p, index=False)

    pm = PlasmaMonitor(PlasmaMonitorConfig(state_dim=8))
    pm.reset_equilibrium = MagicMock(wraps=pm.reset_equilibrium)
    evaluate_shot_csv(p, monitor=pm, state_dim=8)
    called_with = pm.reset_equilibrium.call_args[0][0]
    # Te_core[0] == 10.0 identifies row 0 uniquely (Te_core[1] == 9.95)
    assert called_with[0] == pytest.approx(10.0)


def test_evaluate_dir_csv_default_state_dim_is_eight(tmp_path: Path):
    """`evaluate_dir_csv`'s own `state_dim: int = 8` default was never
    exercised via its own default either -- providing an explicit
    monitor_cfg with state_dim=8 while leaving evaluate_dir_csv's own
    state_dim parameter unset isolates its default specifically: a mutated
    default (e.g. 9) would load 9-dim state into an 8-dim-configured
    monitor, causing a genuine crash."""
    t = np.arange(5, dtype=float)
    q = np.array([0.9, 0.95, 0.97, 0.99, 0.98], dtype=float)
    Te_core = 10.0 - 0.05 * t
    ne = np.full_like(t, 1e20)
    df = pd.DataFrame({"time_ms": t, "Te_core": Te_core, "ne": ne, "q_min": q})
    p = tmp_path / "shot.csv"
    df.to_csv(p, index=False)

    result_df = evaluate_dir_csv(tmp_path, monitor_cfg=PlasmaMonitorConfig(state_dim=8))
    assert len(result_df) == 1


def test_evaluate_dir_csv_builds_config_from_custom_state_dim_when_none_given(tmp_path: Path):
    """`cfg = monitor_cfg or PlasmaMonitorConfig(state_dim=state_dim)` was
    mutated to `and` (short-circuits to None when monitor_cfg is falsy) and
    separately to `cfg = None` outright -- both happen to work when
    monitor_cfg is None AND state_dim is left at its default, because
    PlasmaMonitor's own `__init__` has an independent `config or
    PlasmaMonitorConfig()` fallback that silently defaults state_dim to 8
    regardless. The mutation only becomes observable when the CALLER relies
    on evaluate_dir_csv's own state_dim parameter (not its default) to
    configure a monitor without providing monitor_cfg explicitly -- the
    real code must build a config that respects the custom state_dim;
    either mutant discards it and falls back to PlasmaMonitor's own
    default (8), causing a dimension mismatch against the loaded 10-dim
    state."""
    t = np.arange(5, dtype=float)
    q = np.array([0.9, 0.95, 0.97, 0.99, 0.98], dtype=float)
    Te_core = 10.0 - 0.05 * t
    ne = np.full_like(t, 1e20)
    df = pd.DataFrame({"time_ms": t, "Te_core": Te_core, "ne": ne, "q_min": q})
    p = tmp_path / "shot.csv"
    df.to_csv(p, index=False)

    result_df = evaluate_dir_csv(tmp_path, state_dim=10)  # no monitor_cfg
    assert len(result_df) == 1
