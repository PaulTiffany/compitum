import io
from contextlib import redirect_stdout
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
        # `pytest.raises(match=...)` is an unanchored substring search -- a
        # mutation wrapping the whole message in extra text would still
        # match. Assert the exact message instead.
        with pytest.raises(ValueError) as exc_info:
            metric.batch_distance(np.array([[1, 1]]), np.array([0, 0]))
        assert str(exc_info.value) == "self.W should not be None after _update_cholesky()"


def test_update_cholesky_error_recovery_exact_delta_print_and_upper_triangular() -> None:
    """No existing test checks the exact recovered `delta` value, the debug
    prints' exact content, or that the recovered `W` is upper-triangular
    (`lower=False`) -- only that recovery succeeds at all (`delta > 0`).
    Use a non-diagonal `L` (so upper vs lower triangular Cholesky factors
    actually differ) and a `delta` chosen so the `+1e-3` recovery step lands
    on a clean, unclamped value (0.05 - 1e-3... a `-` sign flip, or a huge
    `+1.001` coefficient, would both land somewhere very different)."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=-0.0005)
    metric.L = np.array([[1.0], [1.0]])
    buf = io.StringIO()
    with redirect_stdout(buf):
        metric._update_cholesky()
    assert metric.delta == 0.0005
    assert buf.getvalue() == (
        "Caught LinAlgError. Old delta: -0.0005\n"
        "New delta: 0.0005\n"
        "New metric_matrix: [[1.0005 1.    ]\n"
        " [1.     1.0005]]\n"
    )
    assert metric.W is not None
    assert metric.W[1, 0] == 0.0  # upper-triangular: nothing below the diagonal


def test_update_cholesky_error_recovery_lower_clamp_is_1e_minus_5() -> None:
    """The recovery step's lower clamp (`max(..., 1e-5)`) was never
    exercised at a delta value where it actually binds -- need
    `delta + 1e-3` to land at or below it, which requires starting from a
    negative-enough delta that also still triggers the LinAlgError."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=-0.001)
    metric.L = np.zeros((2, 1))
    metric._update_cholesky()
    assert metric.delta == 1e-5


def test_distance_len_residuals_exactly_rank_stays_default_sigma() -> None:
    """`len(whitened_residuals) > rank` was only ever tested clearly above
    or below `rank`, never exactly equal to it -- where `>` (correct,
    default sigma) and `>=` (mutant, covariance-based sigma) disagree."""
    metric = SymbolicManifoldMetric(D=2, rank=2, delta=1e-3)
    metric.whitened_residuals = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    assert len(metric.whitened_residuals) == metric.rank
    d, sigma = metric.distance(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    assert sigma == 0.1


def test_batch_distance_len_residuals_exactly_rank_stays_default_sigma() -> None:
    metric = SymbolicManifoldMetric(D=2, rank=2, delta=1e-3)
    metric.whitened_residuals = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    assert len(metric.whitened_residuals) == metric.rank
    d_batch, sigma_batch = metric.batch_distance(np.array([[1.0, 1.0]]), np.array([0.0, 0.0]))
    assert np.all(sigma_batch == 0.1)


def test_batch_distance_uses_x_minus_mu_not_plus() -> None:
    """Every existing batch_distance test uses `mu=[0,0]`, where `x - mu`
    and `x + mu` are identical -- use a nonzero `mu` and check the exact
    resulting distance."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=1e-3)
    metric._update_cholesky()
    d_batch, _ = metric.batch_distance(np.array([[3.0, 3.0]]), np.array([1.0, 1.0]))
    expected = np.linalg.norm(metric.W @ np.array([2.0, 2.0]))
    assert np.isclose(d_batch[0], expected)


def test_batch_distance_sigma_squared_clamped_to_zero_not_positive() -> None:
    """A real (LedoitWolf-fitted) covariance is PSD, so the quadratic form
    `wz @ cov @ wz` realistically never goes negative -- the `max(..., 0.0)`
    clamp is only reachable by forcing an indefinite "covariance" directly."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=1e-3)
    metric.whitened_residuals = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    metric._update_cholesky()
    with patch.object(metric.shrink, "fit") as mock_fit:
        mock_fit.return_value.covariance_ = np.array([[-1.0, 0.0], [0.0, -1.0]])
        _, sigma_batch = metric.batch_distance(
            np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([0.0, 0.0])
        )
    assert np.all(sigma_batch == 0.0)


def test_batch_update_spd_d_batch_safe_epsilon_is_1e_minus_8() -> None:
    """Every other test's `d_batch` is well above 1e-8, where the epsilon
    clamp never binds -- use a `d_batch` far below it so the exact epsilon
    value (not just "some small floor") is at stake, observable via the
    exposed `grad_norm` return value."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.array([[1.0], [2.0]])
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (1e6, {})
    grad_norm = metric.batch_update_spd(
        np.array([[2.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        beta_d=1.0,
        d_batch=np.array([1e-10]),
        eta=1e-9,
        srmf_controller=srmf_controller,
    )
    assert np.isclose(grad_norm, 4e8)


def test_batch_update_spd_grad_norm_arithmetic_exact() -> None:
    """`A_batch`'s `beta_d / (2 * d_safe)` and `grad_L`'s `2 * sum(...)`
    coefficients were never pinned to exact values -- a tiny `eta` isolates
    `grad_norm` from any backtracking/stability-cap interference."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.array([[1.0], [2.0]])
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (1e6, {})
    grad_norm = metric.batch_update_spd(
        np.array([[2.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        beta_d=1.0,
        d_batch=np.array([2.0]),
        eta=1e-9,
        srmf_controller=srmf_controller,
    )
    assert grad_norm == 2.0


def test_batch_update_spd_lipschitz_eta_stab_and_z_norm2_exact() -> None:
    """`z_norm2_batch` (`z*z` vs `z/z`), `lipschitz` (`*`/epsilon), and
    `eta_stab` (`1.0/lipschitz`) were never pinned to exact values -- a huge
    `eta`/`eta_cap` make `eta_stab` the sole binding constraint, so the
    exact `self.L` displacement reveals it directly."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.array([[1.0], [2.0]])
    L_before = metric.L.copy()
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (1e6, {})
    metric.batch_update_spd(
        np.array([[3.0, 4.0]]),
        np.array([[1.0, 1.0]]),
        beta_d=1.0,
        d_batch=np.array([2.0]),
        eta=1e6,
        srmf_controller=srmf_controller,
    )
    z = np.array([2.0, 3.0])
    d_safe = max(2.0, 1e-8)
    A = (1.0 / (2 * d_safe)) * np.outer(z, z)
    grad_L = 2 * (A @ L_before)
    expected_eta_stab = 1.0 / max(1.0 * float(np.sum(z * z)), 1e-8)
    delta_L = L_before - metric.L
    assert np.isclose(delta_L[0, 0] / grad_L[0, 0], expected_eta_stab)
    assert np.isclose(delta_L[1, 0] / grad_L[1, 0], expected_eta_stab)


def test_batch_update_spd_gradient_descent_direction_decreases_L() -> None:
    """`new_L = self.L - eta_eff * grad_L` -- a `-` -> `+` sign flip would
    move `L` in the energy-*increasing* direction. A tiny `eta` isolates
    the direction from backtracking/stability-cap interference."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.array([[1.0], [2.0]])
    L_before = metric.L.copy()
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (1e6, {})
    metric.batch_update_spd(
        np.array([[2.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        beta_d=1.0,
        d_batch=np.array([2.0]),
        eta=1e-9,
        srmf_controller=srmf_controller,
    )
    assert metric.L[0, 0] < L_before[0, 0]


def test_batch_update_spd_backtracking_engages_and_arithmetic_is_exact() -> None:
    """The backtracking block (`if e1 > e0: ... eta_eff *= 0.5 ...`) is
    never exercised by any existing test -- the "stability cap" `eta_stab`
    is mathematically safe (non-overshooting) for a *uniform* batch, since
    it's derived from the exact Lipschitz constant of this quadratic
    surrogate. It's only an *average*-based estimate across samples, though,
    so a batch with wildly differing per-sample magnitudes can still
    overshoot -- forcing real backtracking. This scenario resolves in
    exactly one halving, so `eta_eff *= 0.5` and `new_L = self.L - eta_eff *
    grad_L` are both pinned via the exact resulting `self.L`."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.array([[3.0], [4.0]])
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (1e6, {})
    x_batch = np.array([[100.0, 100.0], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]])
    mu_batch = np.zeros((4, 2))
    metric.batch_update_spd(
        x_batch,
        mu_batch,
        beta_d=0.5,
        d_batch=np.array([1.0, 1.0, 1.0, 1.0]),
        eta=100.0,
        srmf_controller=srmf_controller,
    )
    assert np.allclose(metric.L, np.array([[-4.0], [-3.0]]))


def test_batch_update_spd_fnorm_clamp_between_10_and_11() -> None:
    """`fnorm > 10.0` was never tested with `fnorm` strictly between 10 and
    11 -- exactly at `fnorm == 10.0`, the clamp (`*= 10.0/fnorm`) is a
    mathematical no-op regardless of `>` vs `>=`, so that boundary can't
    distinguish a `10.0` -> `11.0` mutation. `eta=0.0` makes `new_L ==
    self.L` exactly (skips backtracking entirely, `e1 == e0`), isolating
    the clamp check on a precisely-controlled starting `L`."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.array([[10.5], [0.0]])  # Frobenius norm exactly 10.5
    metric._update_cholesky()
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (1.0, {})
    metric.batch_update_spd(
        np.array([[1.0, 1.0]]),
        np.array([[0.0, 0.0]]),
        beta_d=0.1,
        d_batch=np.array([1.0]),
        eta=0.0,
        srmf_controller=srmf_controller,
    )
    assert np.isclose(np.linalg.norm(metric.L, "fro"), 10.0)


def test_batch_update_spd_residual_queue_pops_oldest_first() -> None:
    """The pruning loop (`while len(...) > 100: whitened_residuals.pop(0)`)
    was only ever checked for the resulting *length*, never that it removes
    from the *front* (oldest first, FIFO) rather than some other index."""
    metric = SymbolicManifoldMetric(D=2, rank=1, delta=0.1)
    metric.L = np.array([[0.01], [0.01]])
    metric._update_cholesky()
    metric.whitened_residuals = [np.array([float(i), float(i)]) for i in range(101)]
    srmf_controller = MagicMock()
    srmf_controller.update.return_value = (100.0, {})
    metric.batch_update_spd(
        np.array([[1.0, 1.0]]),
        np.array([[0.0, 0.0]]),
        beta_d=0.1,
        d_batch=np.array([1.0]),
        eta=1e-4,
        srmf_controller=srmf_controller,
    )
    assert len(metric.whitened_residuals) == 100
    # Started at [0, 1, ..., 100] (101 entries) plus 1 newly appended (102
    # total), pruned by 2 pops from the front -> [2, 3, ...] remains.
    assert metric.whitened_residuals[0][0] == 2.0
