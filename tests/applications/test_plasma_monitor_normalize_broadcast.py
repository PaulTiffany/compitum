import numpy as np

from compitum.applications import PlasmaMonitor


def test_normalize_broadcast_scalar_scale():
    dim = 5
    # Pass scales as a scalar array (shape (1,)) to force broadcast branch
    pm = PlasmaMonitor(state_dim=dim, rank=3, curvature_alarm=1e6, scales=np.array([2.0]))
    s0 = np.arange(dim, dtype=float)
    out = pm.ingest_profile(s0, t=0.0)
    assert isinstance(out["alarm_status"], bool)
    assert out["confinement_distance"] >= 0.0

