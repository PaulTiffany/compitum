import math
from typing import List

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from compitum.applications import PlasmaMonitor


@st.composite
def states(draw, dim: int):
    vals = draw(st.lists(st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False), min_size=dim, max_size=dim))
    return np.array(vals, dtype=float)


@given(dim=st.integers(min_value=4, max_value=12),
       s=st.lists(st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False), min_size=4, max_size=12))
@settings(deadline=None)
def test_ingest_returns_finite_and_bounded(dim: int, s: List[float]):
    # Trim/pad s to dim
    s_vec = np.array((s + [0.0]*dim)[:dim], dtype=float)
    pm = PlasmaMonitor(state_dim=dim, rank=min(4, dim))

    out = pm.ingest_profile(s_vec, t=0.0)
    # Finite fields
    for k in ("confinement_distance", "confinement_distance_std", "curvature_signal", "trust_radius", "timestamp_ms"):
        assert k in out
        assert math.isfinite(float(out[k]))
    assert isinstance(out["alarm_status"], bool)
    # Trust radius bounded by controller
    assert 0.2 <= out["trust_radius"] <= 5.0


@given(dim=st.integers(min_value=4, max_value=10), s0=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=4, max_size=10))
@settings(deadline=None)
def test_reset_equilibrium_drives_distance_to_zero(dim: int, s0: List[float]):
    s0v = np.array((s0 + [0.0]*dim)[:dim], dtype=float)
    pm = PlasmaMonitor(state_dim=dim, rank=min(4, dim))
    _ = pm.ingest_profile(s0v, t=0.0)

    # Move away then reset
    s1v = s0v + 1.0
    out1 = pm.ingest_profile(s1v, t=1.0)
    assert out1["confinement_distance"] >= 0.0

    pm.reset_equilibrium(s1v)
    out2 = pm.ingest_profile(s1v, t=2.0)
    assert out2["confinement_distance"] == 0.0

