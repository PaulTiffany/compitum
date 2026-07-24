"""Hand-built sequences with known binding-within-horizon outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.capabilities import Capabilities
from compitum.constraint_oracle.horizon import (
    SequenceStepResult,
    compute_horizon_targets,
)
from compitum.constraint_oracle.static import compute_constraint_pressure
from compitum.models import Model

A = np.eye(4)
B = np.array([2.0, 2.0, 0.0, 0.0])
ALL_REGIONS = {"US", "CA", "EU"}


def _model(name: str) -> Model:
    return Model(
        name=name,
        center=np.zeros(2),
        capabilities=Capabilities(regions=ALL_REGIONS, tools_allowed={"none"}),
        cost=0.1,
    )


def _sequence(xb_by_step) -> list:
    models = [_model("only")]
    utilities = {"only": 1.0}
    return [
        SequenceStepResult(
            step=t, pressure=compute_constraint_pressure(xb, A, B, models, utilities)
        )
        for t, xb in enumerate(xb_by_step)
    ]


def test_binding_within_horizon_and_time_to_binding() -> None:
    # Constraint 0 slack: 2.0, 1.0, -0.5 (binds at step 2), 0.0
    xb_by_step = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([2.5, 0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0, 0.0]),
    ]
    steps = _sequence(xb_by_step)
    horizon_results = compute_horizon_targets(steps, horizon=2)

    # At step 0, constraint 0 binds within [0,2] -> at offset 2 (step 2).
    r0 = horizon_results[0][0]
    assert r0.binding_within_horizon is True
    assert r0.time_to_binding == 2
    assert r0.realized_future_slack == pytest.approx(-0.5)

    # At step 1, window is [1,3]: binds at step 2, offset 1.
    r1 = horizon_results[1][0]
    assert r1.binding_within_horizon is True
    assert r1.time_to_binding == 1

    # At step 2 itself: already binding at offset 0.
    r2 = horizon_results[2][0]
    assert r2.binding_within_horizon is True
    assert r2.time_to_binding == 0

    # At step 3 (last step), window clamps to [3,3]: not violated there.
    r3 = horizon_results[3][0]
    assert r3.binding_within_horizon is False
    assert r3.time_to_binding is None
    assert r3.realized_future_slack == pytest.approx(0.0)


def test_never_binds_within_horizon() -> None:
    xb_by_step = [np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(5)]
    steps = _sequence(xb_by_step)
    horizon_results = compute_horizon_targets(steps, horizon=3)
    for row in horizon_results:
        for target in row:
            assert target.binding_within_horizon is False
            assert target.time_to_binding is None
            assert target.realized_future_slack == pytest.approx(
                2.0
            ) or target.realized_future_slack == pytest.approx(0.0)


def test_zero_horizon_only_looks_at_current_step() -> None:
    xb_by_step = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([2.5, 0.0, 0.0, 0.0]),
    ]
    steps = _sequence(xb_by_step)
    horizon_results = compute_horizon_targets(steps, horizon=0)
    assert horizon_results[0][0].binding_within_horizon is False
    assert horizon_results[1][0].binding_within_horizon is True
    assert horizon_results[1][0].time_to_binding == 0


def test_horizon_result_to_dict() -> None:
    xb_by_step = [np.array([0.0, 0.0, 0.0, 0.0])]
    steps = _sequence(xb_by_step)
    horizon_results = compute_horizon_targets(steps, horizon=0)
    payload = horizon_results[0][0].to_dict()
    assert payload["binding_within_horizon"] is False
    assert payload["horizon"] == 0


def test_empty_sequence_returns_empty() -> None:
    assert compute_horizon_targets([], horizon=5) == []


def test_negative_horizon_is_rejected() -> None:
    xb_by_step = [np.array([0.0, 0.0, 0.0, 0.0])]
    steps = _sequence(xb_by_step)
    with pytest.raises(ValueError, match="non-negative"):
        compute_horizon_targets(steps, horizon=-1)
