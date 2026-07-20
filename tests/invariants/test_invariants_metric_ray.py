import numpy as np
from hypothesis import given, strategies as st

from compitum.metric import SymbolicManifoldMetric


@given(D=st.integers(min_value=8, max_value=64), rank=st.integers(min_value=2, max_value=16))
def test_distance_increases_along_ray(D: int, rank: int) -> None:
    rank = min(rank, D)
    met = SymbolicManifoldMetric(D, rank, delta=1e-3)
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(D)
    v = rng.standard_normal(D)
    v /= np.linalg.norm(v) + 1e-9
    # Increasing radii
    r = [0.0, 0.2, 0.5, 1.0]
    ds = [met.distance(mu + t * v, mu)[0] for t in r]
    # Monotone non-decreasing and strictly increasing beyond tiny tolerance
    assert all(ds[i] <= ds[i + 1] + 1e-12 for i in range(len(ds) - 1))
    assert ds[0] <= ds[-1] + 1e-12
