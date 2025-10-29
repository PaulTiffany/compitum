import numpy as np
from hypothesis import given, strategies as st

from compitum.metric import SymbolicManifoldMetric


@given(D=st.integers(min_value=4, max_value=48), rank=st.integers(min_value=2, max_value=12))
def test_triangle_inequality_under_spd_whitening(D: int, rank: int) -> None:
    rank = min(rank, D)
    met = SymbolicManifoldMetric(D, rank, delta=1e-3)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(D)
    y = rng.standard_normal(D)
    z = rng.standard_normal(D)

    d_xy, _ = met.distance(x, y)
    d_yz, _ = met.distance(y, z)
    d_xz, _ = met.distance(x, z)

    # Triangle inequality with small numerical tolerance
    assert d_xz <= d_xy + d_yz + 1e-9

