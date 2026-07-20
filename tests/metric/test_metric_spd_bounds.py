import numpy as np
from hypothesis import given, strategies as st

from compitum.metric import SymbolicManifoldMetric


@given(D=st.integers(min_value=4, max_value=64), rank=st.integers(min_value=2, max_value=16))
def test_metric_matrix_spd_and_min_eig_ge_delta(D: int, rank: int) -> None:
    rank = min(rank, D)
    met = SymbolicManifoldMetric(D, rank, delta=1e-3)
    M = met.metric_matrix()
    # Symmetric positive definite: eigenvalues > 0
    w = np.linalg.eigvalsh(M)
    assert np.all(w > 0.0)
    # Lower bounded by approximately delta (account for low-rank L term)
    assert w.min() >= met.delta - 1e-6
