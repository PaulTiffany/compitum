import numpy as np
from hypothesis import given, strategies as st

from compitum.coherence import CoherenceFunctional
from compitum.metric import SymbolicManifoldMetric


@given(d=st.integers(min_value=4, max_value=24), rank=st.integers(min_value=2, max_value=12))
def test_negative_log_evidence_correlates_with_metric_distance(d: int, rank: int) -> None:
    rank = min(rank, d)
    rng = np.random.default_rng(0)
    # Fit an isotropic cloud near the origin
    coh = CoherenceFunctional(k=2000)
    for _ in range(600):
        x = rng.normal(0.0, 0.6, size=d)
        coh.update("fast", x, success=1.0)

    met = SymbolicManifoldMetric(d, rank, delta=1e-3)
    mu = np.zeros(d)
    # Sample points at growing radii and compare metrics
    radii = np.linspace(0.0, 1.6, 9)
    vals = []
    v = rng.normal(0.0, 1.0, size=d)
    v /= (np.linalg.norm(v) + 1e-9)
    for r in radii:
        x = r * v
        d_m, _ = met.distance(x, mu)
        l = coh.log_evidence("fast", x)
        vals.append((d_m, -l))

    # As radius increases, both d_m and -log_evidence should be non-decreasing on average
    d_seq = [p[0] for p in vals]
    nle_seq = [p[1] for p in vals]
    assert all(d_seq[i] <= d_seq[i + 1] + 1e-9 for i in range(len(d_seq) - 1))
    assert all(nle_seq[i] <= nle_seq[i + 1] + 1e-3 for i in range(len(nle_seq) - 1))
