
from unittest.mock import MagicMock

import numpy as np

from compitum.metric import SymbolicManifoldMetric


def test_metric_update_cholesky_linalg_error() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.0)
    # Force L to be zero, so L @ L.T is zero matrix, which is not positive definite
    metric.L = np.zeros((2, 1))
    # This should not raise LinAlgError, but handle it by increasing delta
    print(f"Initial delta: {metric.delta}")
    print(f"Initial metric_matrix: {metric.metric_matrix()}")
    try:
        metric._update_cholesky()
    except Exception as e:
        print(f"Exception caught in test: {e}")
        raise
    print(f"Final delta: {metric.delta}")
    print(f"Final metric_matrix: {metric.metric_matrix()}")
    assert metric.delta > 0.0

def test_metric_distance_with_covariance() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1)
    # with rank=1, 2 residuals is enough
    metric.whitened_residuals = [np.array([1,1]), np.array([2,2])]
    d, sigma = metric.distance(np.array([1,1]), np.array([0,0]))
    assert sigma < 0.1 # should be different from the default

def test_metric_update_spd() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric._update_cholesky() # initialize W
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (0.1, {}) # eta_cap, drift

    L_before = metric.L.copy()
    grad_norm = metric.update_spd(
        x=np.array([1,1]),
        mu=np.array([0,0]),
        beta_d=0.5,
        d=1.0,
        eta=0.01,
        srmf_controller=srmf_controller
    )

    assert grad_norm > 0
    assert not np.allclose(L_before, metric.L)
    assert len(metric.whitened_residuals) == 1
    srmf_controller.update.assert_called()

def test_metric_update_spd_large_fnorm() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.ones((2,1)) * 100 # large frobenius norm
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (0.1, {})

    metric.update_spd(np.array([1,1]), np.array([0,0]), 0.5, 1.0, 0.01, srmf_controller)

    fnorm = np.linalg.norm(metric.L, "fro")
    assert np.isclose(fnorm, 10.0)

def test_metric_whitened_residuals_pop() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1)
    metric.whitened_residuals = [np.array([i,i]) for i in range(101)]
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (0.1, {})

    metric.update_spd(np.array([1,1]), np.array([0,0]), 0.5, 1.0, 0.01, srmf_controller)

    assert len(metric.whitened_residuals) == 101
