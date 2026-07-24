"""Declared channel-vector mapping: shape, semantics, and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.constraint_oracle.channels import (
    CHANNEL_DIMENSION,
    ChannelPreviousState,
    compute_channel_vector,
    compute_sequence_channels,
)
from compitum.constraint_oracle.dataset import generate_controlled_dataset
from compitum.constraint_oracle.static import compute_constraint_pressure
from compitum.control import LyapunovController

MODEL_NAMES = ("current", "suppressed", "filler")


def _pressures_and_utilities(seq):
    pressures = []
    utilities_by_step = []
    for case in seq.cases:
        pressure = compute_constraint_pressure(
            case.xB, seq.A, seq.b, seq.models, case.utilities, context=case.context
        )
        pressures.append(pressure)
        utilities_by_step.append(case.utilities)
    return pressures, utilities_by_step


def test_channel_vector_has_declared_dimension_and_is_finite() -> None:
    sequences = generate_controlled_dataset(
        seed=61,
        sequences_per_scenario=2,
        steps_per_sequence=6,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    seq = sequences[0]
    pressures, utilities_by_step = _pressures_and_utilities(seq)
    controller = LyapunovController()
    previous = None
    for pressure, utilities in zip(pressures, utilities_by_step):
        vector, previous = compute_channel_vector(
            pressure, utilities, controller, MODEL_NAMES, previous
        )
        assert vector.shape == (CHANNEL_DIMENSION,)
        assert np.all(np.isfinite(vector))


def test_slack_channels_are_normalized_and_clipped() -> None:
    sequences = generate_controlled_dataset(
        seed=67,
        sequences_per_scenario=1,
        steps_per_sequence=6,
        scenarios=["permanently_infeasible"],
    )
    seq = sequences[0]
    pressures, utilities_by_step = _pressures_and_utilities(seq)
    controller = LyapunovController()
    vector, _ = compute_channel_vector(
        pressures[0], utilities_by_step[0], controller, MODEL_NAMES, None
    )
    slack_channels = vector[0:4]
    assert np.all(slack_channels >= -1.0) and np.all(slack_channels <= 1.0)


def test_feasibility_mask_marks_only_selected_model_when_linear_feasible() -> None:
    sequences = generate_controlled_dataset(
        seed=71,
        sequences_per_scenario=1,
        steps_per_sequence=1,
        scenarios=["permanently_slack"],
    )
    seq = sequences[0]
    pressure = compute_constraint_pressure(
        seq.cases[0].xB, seq.A, seq.b, seq.models, seq.cases[0].utilities
    )
    controller = LyapunovController()
    vector, _ = compute_channel_vector(
        pressure, seq.cases[0].utilities, controller, MODEL_NAMES, None
    )
    mask = vector[4:7]
    assert mask.sum() == 1.0
    winner_index = MODEL_NAMES.index(pressure.selected_model)
    assert mask[winner_index] == 1.0


def test_feasibility_mask_all_zero_when_linear_infeasible() -> None:
    sequences = generate_controlled_dataset(
        seed=73,
        sequences_per_scenario=1,
        steps_per_sequence=1,
        scenarios=["permanently_infeasible"],
    )
    seq = sequences[0]
    pressure = compute_constraint_pressure(
        seq.cases[0].xB, seq.A, seq.b, seq.models, seq.cases[0].utilities
    )
    controller = LyapunovController()
    vector, _ = compute_channel_vector(
        pressure, seq.cases[0].utilities, controller, MODEL_NAMES, None
    )
    assert np.all(vector[4:7] == 0.0)


def test_utility_gap_and_entropy_channels() -> None:
    from compitum.constraint_oracle.static import compute_constraint_pressure as ccp

    utilities = {"current": 1.0, "suppressed": 3.0, "filler": 0.5}
    xB = np.zeros(4)
    A = np.eye(4)
    b = np.array([2.0, 2.0, 0.0, 0.0])
    from compitum.capabilities import Capabilities
    from compitum.models import Model

    models = [
        Model(name=n, center=np.zeros(2), capabilities=Capabilities({"US"}, set()), cost=0.1)
        for n in MODEL_NAMES
    ]
    pressure = ccp(xB, A, b, models, utilities)
    controller = LyapunovController()
    vector, _ = compute_channel_vector(pressure, utilities, controller, MODEL_NAMES, None)
    assert vector[10] == pytest.approx(3.0 - 1.0)  # gap = top - runner-up
    assert vector[11] > 0.0  # entropy of a non-degenerate distribution


def test_controller_drift_state_evolves_across_steps() -> None:
    sequences = generate_controlled_dataset(
        seed=79,
        sequences_per_scenario=1,
        steps_per_sequence=8,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    seq = sequences[0]
    pressures, utilities_by_step = _pressures_and_utilities(seq)
    controller = LyapunovController()
    vectors = []
    previous = None
    for pressure, utilities in zip(pressures, utilities_by_step):
        vector, previous = compute_channel_vector(
            pressure, utilities, controller, MODEL_NAMES, previous
        )
        vectors.append(vector)
    drift_integrals = [v[14] for v in vectors]
    # The ramp eventually violates a constraint, driving d_star > 0, so the
    # drift_integral (which only ever accumulates then decays) must change.
    assert len(set(drift_integrals)) > 1


def test_transition_indicators_zero_on_first_step() -> None:
    sequences = generate_controlled_dataset(
        seed=83,
        sequences_per_scenario=1,
        steps_per_sequence=4,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    seq = sequences[0]
    pressure = compute_constraint_pressure(
        seq.cases[0].xB, seq.A, seq.b, seq.models, seq.cases[0].utilities
    )
    controller = LyapunovController()
    vector, previous = compute_channel_vector(
        pressure, seq.cases[0].utilities, controller, MODEL_NAMES, None
    )
    assert vector[15] == 0.0
    assert vector[16] == 0.0
    assert previous.selected_model == pressure.selected_model


def test_violated_set_transition_detects_a_change() -> None:
    sequences = generate_controlled_dataset(
        seed=89,
        sequences_per_scenario=3,
        steps_per_sequence=8,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    seq = next(
        s for s in sequences if s.known_binding_step is not None and 0 < s.known_binding_step < 7
    )
    pressures, utilities_by_step = _pressures_and_utilities(seq)
    controller = LyapunovController()
    previous = None
    transitions = []
    for pressure, utilities in zip(pressures, utilities_by_step):
        vector, previous = compute_channel_vector(
            pressure, utilities, controller, MODEL_NAMES, previous
        )
        transitions.append(vector[15])
    # Exactly at the binding step, the violated-constraint set changes from
    # empty to non-empty: a nonzero Jaccard-distance transition.
    assert transitions[seq.known_binding_step] > 0.0


def test_compute_sequence_channels_matches_manual_loop() -> None:
    sequences = generate_controlled_dataset(
        seed=97,
        sequences_per_scenario=1,
        steps_per_sequence=5,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    seq = sequences[0]
    pressures, utilities_by_step = _pressures_and_utilities(seq)

    manual_controller = LyapunovController()
    manual_vectors = []
    previous = None
    for pressure, utilities in zip(pressures, utilities_by_step):
        vector, previous = compute_channel_vector(
            pressure, utilities, manual_controller, MODEL_NAMES, previous
        )
        manual_vectors.append(vector)

    via_helper = compute_sequence_channels(pressures, utilities_by_step, MODEL_NAMES)
    for a, b in zip(manual_vectors, via_helper):
        np.testing.assert_array_equal(a, b)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        compute_sequence_channels([], [{"x": 1.0}], MODEL_NAMES)


def test_channel_previous_state_defaults() -> None:
    state = ChannelPreviousState()
    assert state.violated_indices == set()
    assert state.selected_model is None
