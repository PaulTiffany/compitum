from hypothesis import given, strategies as st
import numpy as np

from compitum.metric import SymbolicManifoldMetric
from compitum.control import LyapunovController


@given(
    D=st.integers(min_value=8, max_value=64),
    rank=st.integers(min_value=2, max_value=16),
)
def test_metric_update_keeps_spd_and_nonincreasing_distance(D: int, rank: int) -> None:
    rank = min(rank, D)
    met = SymbolicManifoldMetric(D, rank, delta=1e-3)
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(D).astype(np.float64)
    x = mu + rng.normal(0.0, 0.5, size=D)

    d0, _ = met.distance(x, mu)
    ctrl = LyapunovController()
    beta_d = 0.5
    # Modest step size; controller will cap further if needed
    met.update_spd(x, mu, beta_d=beta_d, d=d0, eta=0.1, srmf_controller=ctrl)

    # Still SPD: Cholesky must succeed
    W = met._update_cholesky()
    assert W is not None and np.all(np.isfinite(W))

    d1, _ = met.distance(x, mu)
    # Allow tiny numeric wiggle; expect non-increase after a single descent step/backtracking
    assert d1 <= d0 + 1e-8

