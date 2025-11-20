import numpy as np

from compitum.applications import PlasmaMonitor


def test_plasma_monitor_basic_flow():
    pm = PlasmaMonitor(state_dim=8, rank=3, curvature_alarm=0.2)

    # Initialize near equilibrium
    s0 = np.zeros(8, dtype=float)
    out0 = pm.ingest_profile(s0, t=0.0)
    assert set(["confinement_distance","curvature_signal","trust_radius","alarm_status","timestamp_ms"]).issubset(out0.keys())
    assert out0["confinement_distance"] >= 0.0
    assert out0["alarm_status"] in (True, False)

    # Move gradually away (simulate drift toward instability)
    curv_vals = [out0["curvature_signal"]]
    alarm_seen = out0["alarm_status"]
    for i in range(1, 25):
        # Decrease synthetic q_min; push state farther each step
        state = np.array([10.0 - 0.05*i, 1e20, 1.5 - 0.01*i, 0, 0, 0, 0, 0], dtype=float)
        out = pm.ingest_profile(state, t=float(i))
        curv_vals.append(out["curvature_signal"])
        alarm_seen = alarm_seen or out["alarm_status"]

    # Curvature should change over time; often increases as drift accumulates
    assert len(set(np.round(v, 6) for v in curv_vals)) >= 2
    assert isinstance(alarm_seen, bool)


def test_plasma_monitor_reset_equilibrium():
    pm = PlasmaMonitor(state_dim=4, rank=2, curvature_alarm=0.1)
    s1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    _ = pm.ingest_profile(s1, t=0.0)

    # Drift
    s2 = np.array([2.0, 0.0, 0.0, 0.0], dtype=float)
    out2 = pm.ingest_profile(s2, t=1.0)
    assert out2["confinement_distance"] > 0

    # Reset center; distance should drop when re-centered
    pm.reset_equilibrium(s2)
    out3 = pm.ingest_profile(s2, t=2.0)
    assert out3["confinement_distance"] == 0.0