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


def test_supercon_kwarg_overrides_take_precedence_over_config():
    base_cfg = SuperconMonitorConfig(state_dim=8, rank=4, alarm_threshold=0.5, norm_p=2.0)
    scales = np.array([2.0, 2.0, 2.0], dtype=float)
    sc = SuperconMonitor(
        base_cfg, state_dim=3, rank=1, alarm_threshold=0.75, scales=scales, norm_p=1.5
    )
    assert sc.metric.D == 3
    assert sc.alarm_threshold == 0.75
    assert sc.norm_p == 1.5
    assert np.array_equal(sc.scales, scales)


def test_supercon_scales_broadcast_shape_mismatch():
    cfg = SuperconMonitorConfig(state_dim=4, rank=2, scales=np.array([2.0], dtype=float))
    sc = SuperconMonitor(cfg)
    x = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
    out = sc.ingest_features(x)
    assert out["distance"] == 0.0  # first call sets center = x, so distance to self is 0


def test_supercon_non_euclidean_norm_p():
    cfg = SuperconMonitorConfig(state_dim=4, rank=2, norm_p=1.0)
    sc = SuperconMonitor(cfg)
    x0 = np.zeros(4, dtype=float)
    _ = sc.ingest_features(x0)
    x1 = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
    out = sc.ingest_features(x1)
    assert out["distance"] > 0

