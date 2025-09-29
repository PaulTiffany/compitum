from typing import Any, Tuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from compitum.control import SRMFController
from compitum.metric import SymbolicManifoldMetric

from .harness import TOL

# --- Strategies for generating test data ---


@st.composite
def metric_params(draw: Any) -> Tuple[int, int]:
    d = draw(st.integers(min_value=2, max_value=16))
    rank = draw(st.integers(min_value=1, max_value=d))
    return d, rank


# Strategy for generating a numpy vector of a given dimension
def vectors(
    dim: int, elements: st.SearchStrategy[float] = st.floats(-1e3, 1e3)
) -> st.SearchStrategy[np.ndarray]:
    return st.lists(elements, min_size=dim, max_size=dim).map(np.array)


# --- Invariant Tests ---


@pytest.mark.invariants
@given(params=metric_params())
def test_metric_spd_properties(params: tuple[int, int]) -> None:
    """
    Tests that the metric matrix M is always Symmetric Positive-Definite.
    POWER: Algebra/Shape
    """
    D, rank = params
    metric = SymbolicManifoldMetric(D, rank)
    M = metric.metric_matrix()

    # 1. Symmetry: M should be equal to its transpose
    assert np.allclose(M, M.T, atol=TOL.abs)

    # 2. Positive-Definite: All eigenvalues of M must be positive
    try:
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)
    except np.linalg.LinAlgError:
        pytest.fail("Metric matrix was not positive definite, leading to LinAlgError")


@pytest.mark.invariants
@given(params=metric_params(), data=st.data())
def test_metric_triangle_inequality(params: tuple[int, int], data: Any) -> None:
    """
    Tests that the metric satisfies the triangle inequality: d(x, z) <= d(x, y) + d(y, z).
    POWER: Algebra/Shape
    """
    D, rank = params
    metric = SymbolicManifoldMetric(D, rank)

    x = data.draw(vectors(dim=D))
    y = data.draw(vectors(dim=D))
    z = data.draw(vectors(dim=D))

    d_xy, _ = metric.distance(x, y)
    d_yz, _ = metric.distance(y, z)
    d_xz, _ = metric.distance(x, z)

    assert d_xz <= d_xy + d_yz + TOL.abs

@pytest.mark.invariants
@given(
    params=metric_params(),
    data=st.data(),
    beta_d=st.floats(-1e3, 1e3),
    d=st.floats(1e-3, 1e3),
    eta=st.floats(1e-3, 1.0),
)
def test_metric_update_stability(
    params: tuple[int, int],
    data: Any,
    beta_d: float,
    d: float,
    eta: float,
) -> None:
    """
    Tests that the Frobenius norm of L is capped during update_spd.
    This ensures the online learning process is stable.
    POWER: Stability/Bounds
    """
    D, rank = params
    metric = SymbolicManifoldMetric(D, rank)
    srmf_controller = SRMFController(kappa=0.1, r0=1.0)

    # Generate a large initial L to ensure the cap is tested
    metric.L = data.draw(st.just(np.random.rand(D, rank) * 20.0))

    x = data.draw(vectors(dim=D))
    mu = data.draw(vectors(dim=D))

    metric.update_spd(
        x=x, mu=mu, beta_d=beta_d, d=d, eta=eta, srmf_controller=srmf_controller
    )

    fnorm = np.linalg.norm(metric.L, "fro")
    assert fnorm <= 10.0 + TOL.abs

@pytest.mark.invariants
@given(params=metric_params(), data=st.data())
def test_metric_whitening_isometry(params: tuple[int, int], data: Any) -> None:
    """
    Tests that the metric distance is the L2 norm in the whitened space.
    d(a, b) == norm(W @ a - W @ b)
    POWER: Algebra/Shape
    """
    D, rank = params
    metric = SymbolicManifoldMetric(D, rank)
    metric._update_cholesky() # Ensure W is calculated

    a = data.draw(vectors(dim=D))
    b = data.draw(vectors(dim=D))

    d_ab, _ = metric.distance(a, b)
    W = metric.W
    assert W is not None
    d_whitened = np.linalg.norm(W @ (a - b))

    assert np.allclose(d_ab, d_whitened, atol=TOL.abs)
