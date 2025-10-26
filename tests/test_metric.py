from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy import linalg

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
    metric.whitened_residuals = [np.array([1, 1]), np.array([2, 2])]
    d, sigma = metric.distance(np.array([1, 1]), np.array([0, 0]))
    assert sigma < 0.1  # should be different from the default


def test_metric_update_spd() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric._update_cholesky()  # initialize W
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (0.1, {})  # eta_cap, drift

    L_before = metric.L.copy()
    grad_norm = metric.update_spd(
        x=np.array([1, 1]),
        mu=np.array([0, 0]),
        beta_d=0.5,
        d=1.0,
        eta=0.01,
        srmf_controller=srmf_controller,
    )

    assert grad_norm > 0
    assert not np.allclose(L_before, metric.L)
    assert len(metric.whitened_residuals) == 1
    srmf_controller.update.assert_called()


def test_metric_update_spd_large_fnorm() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.ones((2, 1)) * 100  # large frobenius norm
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (0.1, {})

    metric.update_spd(np.array([1, 1]), np.array([0, 0]), 0.5, 1.0, 0.01, srmf_controller)

    fnorm = np.linalg.norm(metric.L, "fro")
    assert np.isclose(fnorm, 10.0)


def test_metric_whitened_residuals_pop() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1)
    metric.whitened_residuals = [np.array([i, i]) for i in range(101)]
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (0.1, {})

    metric.update_spd(np.array([1, 1]), np.array([0, 0]), 0.5, 1.0, 0.01, srmf_controller)

    assert len(metric.whitened_residuals) == 100


def test_batch_distance() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1)
    metric.whitened_residuals = [np.array([1, 1]), np.array([2, 2])]
    x_batch = np.array([[1, 1], [2, 2], [3, 3]])
    mu = np.array([0, 0])
    d_batch, sigma_batch = metric.batch_distance(x_batch, mu)
    assert d_batch.shape == (3,)
    assert sigma_batch.shape == (3,)
    assert np.all(sigma_batch > 0)  # Check that sigma is calculated

    # Call it again to cover the branch where self.W is not None
    d_batch_2, sigma_batch_2 = metric.batch_distance(x_batch, mu)
    assert d_batch_2.shape == (3,)
    assert sigma_batch_2.shape == (3,)


def test_distance_default_sigma() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1)
    d, sigma = metric.distance(np.array([1, 1]), np.array([0, 0]))
    assert sigma == 0.1


def test_update_spd_zero_distance() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (0.1, {})
    grad_norm = metric.update_spd(
        x=np.array([0, 0]),
        mu=np.array([0, 0]),
        beta_d=0.5,
        d=0.0,
        eta=0.01,
        srmf_controller=srmf_controller,
    )
    assert grad_norm >= 0


def test_batch_distance_default_sigma() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1)
    x_batch = np.array([[1, 1], [2, 2]])
    mu = np.array([0, 0])
    d_batch, sigma_batch = metric.batch_distance(x_batch, mu)
    assert d_batch.shape == (2,)
    assert sigma_batch.shape == (2,)
    assert np.all(sigma_batch == 0.1)


def test_update_cholesky_double_linalg_error() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=-0.01)
    metric.L = np.zeros((2, 1))
    # This should not raise an error, as the delta is corrected.
    metric._update_cholesky()
    assert metric.delta > 0


@patch("scipy.linalg.cholesky", side_effect=[linalg.LinAlgError("err"), linalg.LinAlgError("err")])
def test_update_cholesky_persistent_error(mock_cholesky: MagicMock) -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.0)
    metric.L = np.zeros((2, 1))
    with pytest.raises(linalg.LinAlgError):
        metric._update_cholesky()
    assert mock_cholesky.call_count == 2


@patch("scipy.linalg.cholesky")
def test_update_cholesky_recovers_from_error(mock_cholesky: MagicMock) -> None:
    # Fail once, then succeed
    mock_cholesky.side_effect = [linalg.LinAlgError("Test Error"), np.eye(2)]

    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.0)
    metric.L = np.zeros((2, 1))

    # This should now succeed without raising an error
    metric._update_cholesky()

    assert mock_cholesky.call_count == 2
    assert metric.delta > 0.0
    assert metric.W is not None

def test_batch_distance_raises_error_if_w_is_none() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=1)
    with patch.object(metric, "_update_cholesky") as mock_update:
        mock_update.return_value = None
        with pytest.raises(ValueError, match="self.W should not be None after _update_cholesky()"):
            metric.batch_distance(np.array([[1, 1]]), np.array([0, 0]))
