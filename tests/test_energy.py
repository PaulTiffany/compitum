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

def test_beta_d_property() -> None:
    energy = SymbolicFreeEnergy(1, 1, 1, 1, 1)
    assert energy.beta_d == 1
    energy.beta_d = 0.5
    assert energy.beta_d == 0.5
