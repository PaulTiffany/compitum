import numpy as np

from compitum.metric import SymbolicManifoldMetric


def test_spd_properties() -> None:
    m = SymbolicManifoldMetric(20, 5)
    M = m.metric_matrix()
    assert np.allclose(M, M.T)
    eig = np.linalg.eigvalsh(M)
    assert np.all(eig > 0)

def test_triangle_inequality() -> None:
    m = SymbolicManifoldMetric(12, 4)
    x, y, z = np.random.randn(12), np.random.randn(12), np.random.randn(12)
    d_xy, _ = m.distance(x, y)
    d_yz, _ = m.distance(y, z)
    d_xz, _ = m.distance(x, z)
    assert d_xz <= d_xy + d_yz + 1e-9

def test_whitening_isometry() -> None:
    m = SymbolicManifoldMetric(10, 3)
    m._update_cholesky()
    assert m.W is not None
    a, b = np.random.randn(10), np.random.randn(10)
    d, _ = m.distance(a, b)
    wa, wb = m.W @ a, m.W @ b
    assert np.isclose(d, np.linalg.norm(wa - wb), rtol=1e-9)
