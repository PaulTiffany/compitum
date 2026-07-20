import numpy as np
from hypothesis import given, strategies as st

from compitum.coherence import CoherenceFunctional


@given(d=st.integers(min_value=2, max_value=16))
def test_coherence_monotone_outward_on_isotropic_cloud(d: int) -> None:
    coh = CoherenceFunctional(k=1000)
    rng = np.random.default_rng(0)
    # Fill reservoir for model "fast" with isotropic Gaussian near 0
    for _ in range(400):
        x = rng.normal(0.0, 0.5, size=d).astype(np.float64)
        coh.update("fast", x, success=1.0)

    # Evaluate along a random ray; ensure center has highest evidence
    v = rng.normal(0.0, 1.0, size=d)
    v /= np.linalg.norm(v) + 1e-9
    vals = []
    for r in (0.0, 0.5, 1.0, 1.5):
        vals.append(coh.log_evidence("fast", r * v))

    # Monotone non-increasing as radius grows (allow small numerical jitter)
    assert vals[0] >= vals[-1] - 1e-3
    assert all(vals[i] >= vals[i + 1] - 1e-3 for i in range(len(vals) - 1))
