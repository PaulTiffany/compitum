import math
from typing import List

import numpy as np
from hypothesis import given, strategies as st, settings

from compitum.applications import PlasmaMonitor


@given(
    dim=st.integers(min_value=4, max_value=10),
    scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    state_vals=st.lists(
        st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=10,
    ),
)
@settings(deadline=None)
def test_two_way_street_units_scale_invariance(dim: int, scale: float, state_vals: List[float]):
    s = np.array((state_vals + [0.0] * dim)[:dim], dtype=float)
    # Monitor A: base scales=1
    pmA = PlasmaMonitor(state_dim=dim, rank=min(4, dim), scales=np.ones(dim))
    outA = pmA.ingest_profile(s, t=0.0)

    # Monitor B: state scaled by `scale` and scales scaled identically
    pmB = PlasmaMonitor(state_dim=dim, rank=min(4, dim), scales=np.ones(dim) * scale)
    outB = pmB.ingest_profile(s * scale, t=0.0)

    # Invariance under consistent unit scaling
    assert math.isclose(
        outA["confinement_distance"], outB["confinement_distance"], rel_tol=1e-9, abs_tol=1e-9
    )
    assert math.isclose(
        outA["curvature_signal"], outB["curvature_signal"], rel_tol=1e-9, abs_tol=1e-9
    )
