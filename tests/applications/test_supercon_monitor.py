import numpy as np

from compitum.applications.supercon import SuperconMonitor, SuperconMonitorConfig


def test_supercon_no_alarm_at_equilibrium():
    cfg = SuperconMonitorConfig(state_dim=6, rank=3, alarm_threshold=0.0)
    sc = SuperconMonitor(cfg)
    x = np.zeros(6, dtype=float)
    out = sc.ingest_features(x, t=0.0)
    assert out["distance"] == 0.0
    assert out["pairing_proxy"] == 0.0
    assert out["alarm_status"] is False


def test_supercon_reset_center():
    cfg = SuperconMonitorConfig(state_dim=4, rank=2, alarm_threshold=1.0)
    sc = SuperconMonitor(cfg)
    x0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    _ = sc.ingest_features(x0)
    x1 = np.array([2.0, 0.0, 0.0, 0.0], dtype=float)
    out1 = sc.ingest_features(x1)
    assert out1["distance"] > 0
    sc.reset_center(x1)
    out2 = sc.ingest_features(x1)
    assert out2["distance"] == 0.0

