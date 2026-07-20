from typing import cast
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from compitum.coherence import CoherenceFunctional
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.predictors import CalibratedPredictor


def test_symbolic_free_energy_compute() -> None:
    # 1. Setup Mocks
    mock_metric = MagicMock(spec=SymbolicManifoldMetric)
    mock_metric.distance.return_value = (0.5, 0.1)  # d, d_std
    # Mock the W property, which is a numpy array
    type(mock_metric).W = PropertyMock(return_value=np.eye(2))

    mock_predictor = MagicMock(spec=CalibratedPredictor)
    mock_predictor.predict.return_value = (
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.9]),
    )  # q, q_lo, q_hi
    predictors = {
        "quality": mock_predictor,
        "latency": mock_predictor,
        "cost": mock_predictor,
    }

    mock_coherence = MagicMock(spec=CoherenceFunctional)
    mock_coherence.log_evidence.return_value = 0.2

    mock_model = MagicMock(spec=Model)
    mock_model.center = np.zeros(2)
    mock_model.name = "test_model"
    mock_model.cost = 0.0

    xR = np.ones(2)

    # 2. Instantiate and run
    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.2, beta_c=0.1, beta_d=0.5, beta_s=0.3)
    U, U_var, comps = energy.compute(
        xR,
        mock_model,
        cast(dict[str, CalibratedPredictor], predictors),
        mock_coherence,
        mock_metric,
    )

    # 3. Assertions
    # Check that dependencies were called
    mock_metric.distance.assert_called_with(xR, mock_model.center)

    # Check predict calls
    assert mock_predictor.predict.call_count == 3
    predict_call_args, _ = mock_predictor.predict.call_args
    np.testing.assert_array_equal(predict_call_args[0], np.array([xR]))

    # Check the call to log_evidence manually because of numpy array comparison issues
    mock_coherence.log_evidence.assert_called_once()
    call_args, _ = mock_coherence.log_evidence.call_args
    assert call_args[0] == "test_model"
    np.testing.assert_array_equal(call_args[1], np.ones(2))

    # Check calculation of U
    # U = alpha*q - beta_t*t - beta_c*c - beta_d*d + beta_s*log_e
    # U = 1.0*0.8 - 0.2*0.8 - 0.1*0.8 - 0.5*0.5 + 0.3*0.2 = 0.8 - 0.16 - 0.08 - 0.25 + 0.06 = 0.37
    assert np.isclose(U, 0.37)

    # Check components dict
    assert comps["quality"] == 0.8
    assert comps["latency"] == -0.8
    assert comps["cost"] == -0.8
    assert comps["distance"] == -0.5
    assert comps["evidence"] == 0.2

    # Neither U_var (the second return value, already sqrt'd) nor
    # comps["uncertainty"] were checked above -- the entire variance formula
    # (including the /3.92 normal-quantile divisions) was unverified.
    # U_var = (alpha*(q_hi-q_lo)/3.92)^2 + (beta_t*(t_hi-t_lo)/3.92)^2
    #       + (beta_c*(c_hi-c_lo)/3.92)^2 + (beta_d*d_std)^2, then sqrt'd
    expected_uncertainty = (
        (1.0 * (0.9 - 0.7) / 3.92) ** 2
        + (0.2 * (0.9 - 0.7) / 3.92) ** 2
        + (0.1 * (0.9 - 0.7) / 3.92) ** 2
        + (0.5 * 0.1) ** 2
    ) ** 0.5
    assert np.isclose(U_var, expected_uncertainty)
    assert np.isclose(comps["uncertainty"], expected_uncertainty)


def test_symbolic_free_energy_compute_cost_sign_is_addition() -> None:
    # The test above uses model.cost=0.0, where `c[0] + model.cost` and
    # `c[0] - model.cost` are indistinguishable -- a sign-flip mutation on
    # that term would survive undetected. Use a nonzero cost so the sign is
    # actually observable in comps["cost"] and U.
    mock_metric = MagicMock(spec=SymbolicManifoldMetric)
    mock_metric.distance.return_value = (0.5, 0.1)
    type(mock_metric).W = PropertyMock(return_value=np.eye(2))

    mock_predictor = MagicMock(spec=CalibratedPredictor)
    mock_predictor.predict.return_value = (
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.9]),
    )
    predictors = {
        "quality": mock_predictor,
        "latency": mock_predictor,
        "cost": mock_predictor,
    }

    mock_coherence = MagicMock(spec=CoherenceFunctional)
    mock_coherence.log_evidence.return_value = 0.2

    mock_model = MagicMock(spec=Model)
    mock_model.center = np.zeros(2)
    mock_model.name = "test_model"
    mock_model.cost = 0.15

    xR = np.ones(2)
    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.2, beta_c=0.1, beta_d=0.5, beta_s=0.3)
    U, _, comps = energy.compute(
        xR,
        mock_model,
        cast(dict[str, CalibratedPredictor], predictors),
        mock_coherence,
        mock_metric,
    )

    assert comps["cost"] == -(0.8 + 0.15)
    expected_U = 1.0 * 0.8 - 0.2 * 0.8 - 0.1 * (0.8 + 0.15) - 0.5 * 0.5 + 0.3 * 0.2
    assert np.isclose(U, expected_U)


def test_beta_d_property() -> None:
    energy = SymbolicFreeEnergy(1, 1, 1, 1, 1)
    assert energy.beta_d == 1
    energy.beta_d = 0.5
    assert energy.beta_d == 0.5


def test_symbolic_free_energy_batch_compute() -> None:
    # 1. Setup Mocks
    mock_metric = MagicMock(spec=SymbolicManifoldMetric)
    mock_metric.batch_distance.return_value = (np.array([0.5, 0.6]), np.array([0.1, 0.1]))
    type(mock_metric).W = PropertyMock(return_value=np.eye(2))

    mock_predictor = MagicMock(spec=CalibratedPredictor)
    mock_predictor.predict.return_value = (
        np.array([0.8, 0.85]),
        np.array([0.7, 0.75]),
        np.array([0.9, 0.95]),
    )
    predictors = {
        "quality": mock_predictor,
        "latency": mock_predictor,
        "cost": mock_predictor,
    }

    mock_coherence = MagicMock(spec=CoherenceFunctional)
    mock_coherence.batch_log_evidence.return_value = np.array([0.2, 0.25])

    mock_model = MagicMock(spec=Model)
    mock_model.center = np.zeros(2)
    mock_model.name = "test_model"
    mock_model.cost = 0.0

    xR_batch = np.ones((2, 2))

    # 2. Instantiate and run
    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.2, beta_c=0.1, beta_d=0.5, beta_s=0.3)
    U_batch, U_var_batch, comps_list = energy.batch_compute(
        xR_batch,
        mock_model,
        cast(dict[str, CalibratedPredictor], predictors),
        mock_coherence,
        mock_metric,
    )

    # 3. Assertions
    assert U_batch.shape == (2,)
    assert U_var_batch.shape == (2,)
    assert len(comps_list) == 2
    mock_metric.batch_distance.assert_called_once()
    mock_coherence.batch_log_evidence.assert_called_once()
    assert mock_predictor.predict.call_count == 3

    # Check calculation for the first element
    # U = 1.0*0.8 - 0.2*0.8 - 0.1*0.8 - 0.5*0.5 + 0.3*0.2 = 0.37
    assert np.isclose(U_batch[0], 0.37)
    assert comps_list[0]["quality"] == 0.8


def test_batch_compute_center_sign_cost_sign_and_comps_values() -> None:
    # The test above uses model.center=zeros and model.cost=0.0, where both
    # `xR_batch - center` vs `+ center` and `c + cost` vs `c - cost` are
    # unobservable, and only comps_list[0]["quality"] is ever checked (leaving
    # latency/cost/distance/evidence/uncertainty keys and signs, and the
    # entire U_var_batch formula, unverified). Use nonzero center/cost and
    # assert every comps_list key plus the exact uncertainty value.
    mock_metric = MagicMock(spec=SymbolicManifoldMetric)
    mock_metric.batch_distance.return_value = (np.array([0.5]), np.array([0.1]))
    type(mock_metric).W = PropertyMock(return_value=np.eye(1))

    mock_predictor = MagicMock(spec=CalibratedPredictor)
    mock_predictor.predict.return_value = (
        np.array([0.8]),
        np.array([0.7]),
        np.array([0.9]),
    )
    predictors = {
        "quality": mock_predictor,
        "latency": mock_predictor,
        "cost": mock_predictor,
    }

    mock_coherence = MagicMock(spec=CoherenceFunctional)
    mock_coherence.batch_log_evidence.return_value = np.array([0.2])

    mock_model = MagicMock(spec=Model)
    mock_model.center = np.array([3.0])
    mock_model.name = "m"
    mock_model.cost = 0.15

    xR_batch = np.array([[5.0]])
    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.2, beta_c=0.1, beta_d=0.5, beta_s=0.3)
    U_batch, U_var_batch, comps_list = energy.batch_compute(
        xR_batch,
        mock_model,
        cast(dict[str, CalibratedPredictor], predictors),
        mock_coherence,
        mock_metric,
    )

    # xw_batch = W @ (xR_batch - center).T; with W=eye(1), xR=5, center=3,
    # correct is 2.0 -- the `+ center` mutant would instead pass 8.0.
    call_args = mock_coherence.batch_log_evidence.call_args[0]
    np.testing.assert_allclose(call_args[1], np.array([[2.0]]))

    assert comps_list[0]["quality"] == 0.8
    assert comps_list[0]["latency"] == -0.8
    assert comps_list[0]["cost"] == -(0.8 + 0.15)
    assert comps_list[0]["distance"] == -0.5
    assert comps_list[0]["evidence"] == 0.2

    expected_U = 1.0 * 0.8 - 0.2 * 0.8 - 0.1 * (0.8 + 0.15) - 0.5 * 0.5 + 0.3 * 0.2
    assert np.isclose(U_batch[0], expected_U)

    expected_uncertainty = (
        (1.0 * (0.9 - 0.7) / 3.92) ** 2
        + (0.2 * (0.9 - 0.7) / 3.92) ** 2
        + (0.1 * (0.9 - 0.7) / 3.92) ** 2
        + (0.5 * 0.1) ** 2
    ) ** 0.5
    assert np.isclose(U_var_batch[0], expected_uncertainty)
    assert np.isclose(comps_list[0]["uncertainty"], expected_uncertainty)


def test_compute_and_batch_with_none_w() -> None:
    # Setup mocks where W is initially None
    mock_metric = MagicMock(spec=SymbolicManifoldMetric)
    type(mock_metric).W = PropertyMock(side_effect=[None, np.eye(2)])
    mock_metric._update_cholesky.return_value = np.eye(2)
    mock_metric.distance.return_value = (0.5, 0.1)
    mock_metric.batch_distance.return_value = (np.array([0.5]), np.array([0.1]))

    mock_predictor = MagicMock(spec=CalibratedPredictor)
    mock_predictor.predict.return_value = (np.array([0.8]), np.array([0.7]), np.array([0.9]))
    predictors = {"quality": mock_predictor, "latency": mock_predictor, "cost": mock_predictor}
    mock_coherence = MagicMock(spec=CoherenceFunctional)
    mock_coherence.log_evidence.return_value = 0.2
    mock_coherence.batch_log_evidence.return_value = np.array([0.2])
    mock_model = MagicMock(spec=Model)
    mock_model.center = np.zeros(2)
    mock_model.name = "m"
    mock_model.cost = 0.0
    xR = np.ones(2)

    energy = SymbolicFreeEnergy(1, 1, 1, 1, 1)

    # Test compute
    energy.compute(xR, mock_model, predictors, mock_coherence, mock_metric)
    mock_metric._update_cholesky.assert_called_once()

    # Test batch_compute
    mock_metric._update_cholesky.reset_mock()
    type(mock_metric).W = PropertyMock(side_effect=[None, np.eye(2)])  # Reset side_effect
    energy.batch_compute(np.array([xR]), mock_model, predictors, mock_coherence, mock_metric)
    mock_metric._update_cholesky.assert_called_once()


# @patch("builtins.print")
# def test_step_printing(mock_print: MagicMock) -> None:
#     # Re-use mocks from a previous test, simplifying setup
#     mock_metric = MagicMock(spec=SymbolicManifoldMetric)
#     mock_metric.distance.return_value = (0.5, 0.1)
#     type(mock_metric).W = PropertyMock(return_value=np.eye(2))
#     mock_predictor = MagicMock(spec=CalibratedPredictor)
#     mock_predictor.predict.return_value = (np.array([0.8]), np.array([0.7]), np.array([0.9]))
#     predictors = {"quality": mock_predictor, "latency": mock_predictor, "cost": mock_predictor}
#     mock_coherence = MagicMock(spec=CoherenceFunctional)
#     mock_coherence.log_evidence.return_value = 0.2
#     mock_model = MagicMock(spec=Model)
#     mock_model.center = np.zeros(2)
#     mock_model.name = "m"
#     mock_model.cost = 0.0
#     xR = np.ones(2)

#     energy = SymbolicFreeEnergy(1, 1, 1, 1, 1)

#     # Test compute
#     for i in range(100):
#         energy.compute(xR, mock_model, predictors, mock_coherence, mock_metric)
#     mock_print.assert_called_once()
#     assert "SymbolicFreeEnergy.compute" in mock_print.call_args[0][0]

#     # Test batch_compute
#     mock_print.reset_mock()
#     energy = SymbolicFreeEnergy(1, 1, 1, 1, 1)  # Reset energy object to reset step counter
#     mock_metric.batch_distance.return_value = (np.array([0.5] * 50), np.array([0.1] * 50))
#     mock_coherence.batch_log_evidence.return_value = np.array([0.2] * 50)
#     mock_predictor.predict.return_value = (
#         np.array([0.8] * 50),
#         np.array([0.7] * 50),
#         np.array([0.9] * 50),
#     )
#     energy.batch_compute(np.ones((50, 2)), mock_model, predictors, mock_coherence, mock_metric)
#     energy.batch_compute(np.ones((50, 2)), mock_model, predictors, mock_coherence, mock_metric)
#     mock_print.assert_called_once()
#     assert "SymbolicFreeEnergy.batch_compute" in mock_print.call_args[0][0]
