from typing import Dict, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given

from compitum.energy import SymbolicFreeEnergy
from compitum.predictors import CalibratedPredictor

from .harness import TOL, boundary_floats


@pytest.mark.invariants
@given(
    alpha=boundary_floats(0, 1),
    beta_t=boundary_floats(0, 1),
    beta_c=boundary_floats(0, 1),
    beta_d=boundary_floats(0, 1),
    beta_s=boundary_floats(0, 1),
    d=boundary_floats(0, 10),
    log_e=boundary_floats(-10, 10),
)
def test_energy_monotonicity_invariant(
    alpha: float, beta_t: float, beta_c: float, beta_d: float, beta_s: float, d: float, log_e: float
) -> None:
    # POWER: ReplaceBinaryOperator_Add_Sub, ReplaceBinaryOperator_Sub_Add
    # This property checks that the symbolic free energy `U` (a utility) does not
    # increase as a penalty term (like distance `d`) increases.
    energy_func = SymbolicFreeEnergy(alpha, beta_t, beta_c, beta_d, beta_s)

    # Mock dependencies to isolate the term under test
    mock_metric = MagicMock()
    mock_predictors = {
        "quality": MagicMock(spec=CalibratedPredictor),
        "latency": MagicMock(spec=CalibratedPredictor),
        "cost": MagicMock(spec=CalibratedPredictor),
    }
    mock_coherence = MagicMock()
    mock_model = MagicMock()
    mock_model.cost = 0.0

    # Base case
    mock_metric.distance.return_value = (d, 0.1)
    mock_coherence.log_evidence.return_value = log_e
    for p in mock_predictors.values():
        p.predict.return_value = (np.array([0.5]), np.array([0.4]), np.array([0.6]))

    U1, _, _ = energy_func.compute(
        np.zeros(1),
        mock_model,
        cast(Dict[str, CalibratedPredictor], mock_predictors),
        mock_coherence,
        mock_metric,
    )

    # Increased distance should not increase utility
    mock_metric.distance.return_value = (d + 1.0, 0.1)
    U2, _, _ = energy_func.compute(
        np.zeros(1),
        mock_model,
        cast(Dict[str, CalibratedPredictor], mock_predictors),
        mock_coherence,
        mock_metric,
    )

    assert U2 <= U1 + TOL.abs, "Utility should not increase as distance penalty increases"
