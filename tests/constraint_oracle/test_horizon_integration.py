"""Cross-validates horizon.compute_horizon_targets against the controlled
dataset generator's own independently-computed known_binding_step ground
truth -- the actual tranche 2.3 deliverable: horizon targets applied to
real generated sequences, not just hand-built ones.
"""

from __future__ import annotations

from compitum.constraint_oracle.dataset import generate_controlled_dataset
from compitum.constraint_oracle.horizon import (
    SequenceStepResult,
    compute_horizon_targets,
)
from compitum.constraint_oracle.static import compute_constraint_pressure


def _run_sequence(seq, horizon: int):
    steps = [
        SequenceStepResult(
            step=case.step,
            pressure=compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities, context=case.context
            ),
        )
        for case in seq.cases
    ]
    return compute_horizon_targets(steps, horizon=horizon)


def test_time_to_binding_matches_known_binding_step_for_ramp_scenarios() -> None:
    sequences = generate_controlled_dataset(
        seed=41,
        sequences_per_scenario=5,
        steps_per_sequence=8,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    for seq in sequences:
        assert seq.known_binding_step is not None
        i = seq.known_binding_index
        horizon_results = _run_sequence(seq, horizon=len(seq.cases))
        step0 = horizon_results[0][i]
        assert step0.binding_within_horizon is True
        assert step0.time_to_binding == seq.known_binding_step


def test_permanently_slack_never_binds_within_any_horizon() -> None:
    sequences = generate_controlled_dataset(
        seed=43,
        sequences_per_scenario=3,
        steps_per_sequence=6,
        scenarios=["permanently_slack"],
    )
    for seq in sequences:
        assert seq.known_binding_step is None
        horizon_results = _run_sequence(seq, horizon=len(seq.cases))
        for row in horizon_results:
            for target in row:
                assert target.binding_within_horizon is False
                assert target.time_to_binding is None


def test_permanently_infeasible_binds_immediately_at_every_step() -> None:
    sequences = generate_controlled_dataset(
        seed=47,
        sequences_per_scenario=3,
        steps_per_sequence=5,
        scenarios=["permanently_infeasible"],
    )
    for seq in sequences:
        i = seq.known_binding_index
        horizon_results = _run_sequence(seq, horizon=2)
        for row in horizon_results:
            assert row[i].binding_within_horizon is True
            assert row[i].time_to_binding == 0


def test_short_horizon_can_miss_a_binding_event_that_a_long_one_catches() -> None:
    """The single binding event lies outside a short window but inside a
    long one -- confirming the horizon parameter genuinely bounds lookahead
    rather than silently scanning the whole sequence."""
    sequences = generate_controlled_dataset(
        seed=53,
        sequences_per_scenario=6,
        steps_per_sequence=10,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    seq = next(
        s for s in sequences if s.known_binding_step is not None and s.known_binding_step > 1
    )
    i = seq.known_binding_index

    short = _run_sequence(seq, horizon=0)
    long = _run_sequence(seq, horizon=len(seq.cases))

    assert short[0][i].binding_within_horizon is False
    assert long[0][i].binding_within_horizon is True
    assert long[0][i].time_to_binding == seq.known_binding_step
