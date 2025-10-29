import numpy as np
from hypothesis import given, strategies as st

from compitum.coherence import CoherenceFunctional


def _finite_diff_grad(f, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    for i in range(x.shape[0]):
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


@given(d=st.integers(min_value=4, max_value=16))
def test_score_points_inward_on_isotropic_cloud(d: int) -> None:
    rng = np.random.default_rng(0)
    coh = CoherenceFunctional(k=2000)
    # Fit isotropic cloud near origin (already whitened space)
    for _ in range(800):
        coh.update("fast", rng.normal(0.0, 0.6, size=d), success=1.0)

    def loge(x: np.ndarray) -> float:
        return coh.log_evidence("fast", x)

    # Sample a ray and evaluate at moderate radius to avoid flat clipping
    v = rng.normal(0.0, 1.0, size=d)
    v /= (np.linalg.norm(v) + 1e-9)
    x = 0.8 * v
    g = _finite_diff_grad(loge, x, eps=1e-3)
    # Directionality: gradient of log p points inward, so <g, -x> >= 0 within tolerance
    cos = float(np.dot(g, -x) / ((np.linalg.norm(g) + 1e-12) * (np.linalg.norm(x) + 1e-12)))
    assert cos >= -0.05

