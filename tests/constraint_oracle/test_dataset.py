"""Controlled dataset generator: determinism and per-scenario oracle checks."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.constraint_oracle.dataset import (
    SCENARIOS,
    generate_controlled_dataset,
)
from compitum.constraint_oracle.static import compute_constraint_pressure


def test_deterministic_given_seed() -> None:
    a = generate_controlled_dataset(seed=42, sequences_per_scenario=2, steps_per_sequence=4)
    b = generate_controlled_dataset(seed=42, sequences_per_scenario=2, steps_per_sequence=4)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert sa.sequence_id == sb.sequence_id
        assert sa.scenario == sb.scenario
        assert sa.known_binding_step == sb.known_binding_step
        for ca, cb in zip(sa.cases, sb.cases):
            np.testing.assert_array_equal(ca.xB, cb.xB)
            assert ca.utilities == cb.utilities


def test_different_seeds_differ() -> None:
    a = generate_controlled_dataset(seed=1, sequences_per_scenario=1, steps_per_sequence=4)
    b = generate_controlled_dataset(seed=2, sequences_per_scenario=1, steps_per_sequence=4)
    # At least one sequence's utility gap should differ across seeds.
    assert any(
        sa.cases[0].utilities != sb.cases[0].utilities
        for sa, sb in zip(a, b)
        if sa.scenario == sb.scenario
    )


def test_all_scenarios_generated_by_default() -> None:
    sequences = generate_controlled_dataset(seed=7, sequences_per_scenario=1, steps_per_sequence=4)
    seen = {s.scenario for s in sequences}
    assert seen == set(SCENARIOS)


def test_permanently_slack_never_binds() -> None:
    sequences = generate_controlled_dataset(
        seed=3,
        sequences_per_scenario=3,
        steps_per_sequence=6,
        scenarios=["permanently_slack"],
    )
    for seq in sequences:
        assert seq.known_binding_step is None
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities
            )
            assert pressure.feasible is True
            assert all(not t.already_violated for t in pressure.targets)


def test_single_constraint_ramp_recoverable_matches_oracle() -> None:
    sequences = generate_controlled_dataset(
        seed=11,
        sequences_per_scenario=4,
        steps_per_sequence=8,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    for seq in sequences:
        assert seq.known_binding_step is not None
        i = seq.known_binding_index
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities
            )
            target = pressure.targets[i]
            if case.step >= seq.known_binding_step:
                assert target.already_violated is True
                assert target.reason == "recovers_feasibility"
                assert target.best_suppressed_competitor == "suppressed"
                assert target.critical_relaxation is not None
                assert target.critical_relaxation >= 0.0
            else:
                assert target.already_violated is False


def test_capability_blocked_ramp_recovers_via_other_model_not_suppressed() -> None:
    """ "suppressed" is region-blocked, but "current"/"filler" remain
    capable: any current violation blocks every model at once (including
    "current"), so relaxing the sole violated constraint still recovers
    feasibility -- just never via "suppressed" specifically."""
    sequences = generate_controlled_dataset(
        seed=13,
        sequences_per_scenario=3,
        steps_per_sequence=6,
        scenarios=["single_constraint_ramp_capability_blocked"],
    )
    found_recovery = False
    for seq in sequences:
        i = seq.known_binding_index
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities, context=case.context
            )
            target = pressure.targets[i]
            if target.already_violated:
                assert target.reason == "recovers_feasibility"
                assert target.best_suppressed_competitor != "suppressed"
                found_recovery = True
    assert found_recovery


def test_all_capability_blocked_ramp_never_recovers() -> None:
    """Every model in the pool is region-blocked: relaxing the sole
    violated constraint can never recover any pick at all."""
    sequences = generate_controlled_dataset(
        seed=37,
        sequences_per_scenario=3,
        steps_per_sequence=6,
        scenarios=["single_constraint_ramp_all_capability_blocked"],
    )
    found_blocked = False
    for seq in sequences:
        i = seq.known_binding_index
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities, context=case.context
            )
            target = pressure.targets[i]
            if target.already_violated:
                assert target.reason == "capability_blocked_only"
                assert target.critical_relaxation is None
                found_blocked = True
    assert found_blocked


def test_no_higher_utility_ramp_never_recovers_via_suppressed() -> None:
    """ "suppressed" has deliberately low utility; even though relaxing the
    sole violated constraint still recovers "current" (the fallback
    infeasible branch has no utility baseline to beat), it must never
    recover via "suppressed" itself."""
    sequences = generate_controlled_dataset(
        seed=17,
        sequences_per_scenario=3,
        steps_per_sequence=6,
        scenarios=["single_constraint_ramp_no_higher_utility"],
    )
    found_recovery = False
    for seq in sequences:
        i = seq.known_binding_index
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities
            )
            target = pressure.targets[i]
            if target.already_violated:
                assert target.reason == "recovers_feasibility"
                assert target.best_suppressed_competitor == "current"
                found_recovery = True
    assert found_recovery


def test_multi_constraint_joint_reports_blocked_by_other() -> None:
    sequences = generate_controlled_dataset(
        seed=19,
        sequences_per_scenario=3,
        steps_per_sequence=8,
        scenarios=["multi_constraint_joint"],
    )
    found_joint_block = False
    for seq in sequences:
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities
            )
            violated = [t.index for t in pressure.targets if t.already_violated]
            if len(violated) > 1:
                found_joint_block = True
                for t in pressure.targets:
                    if t.already_violated:
                        assert t.reason == "blocked_by_other_constraint"
    assert found_joint_block


def test_discontinuous_tie_scenario_detects_tie() -> None:
    sequences = generate_controlled_dataset(
        seed=23,
        sequences_per_scenario=4,
        steps_per_sequence=8,
        scenarios=["discontinuous_tie"],
    )
    found_tie = False
    for seq in sequences:
        i = seq.known_binding_index
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities
            )
            target = pressure.targets[i]
            if target.already_violated and target.reason == "recovers_feasibility":
                assert target.discontinuous_winner_change is True
                assert set(target.tied_competitors) == {"tie_a", "tie_b"}
                found_tie = True
    assert found_tie


def test_unbinding_recovery_starts_violated_and_ends_feasible() -> None:
    sequences = generate_controlled_dataset(
        seed=29,
        sequences_per_scenario=3,
        steps_per_sequence=6,
        scenarios=["unbinding_recovery"],
    )
    for seq in sequences:
        i = seq.known_binding_index
        first = compute_constraint_pressure(
            seq.cases[0].xB, seq.A, seq.b, seq.models, seq.cases[0].utilities
        )
        last = compute_constraint_pressure(
            seq.cases[-1].xB, seq.A, seq.b, seq.models, seq.cases[-1].utilities
        )
        assert first.targets[i].already_violated is True
        assert last.targets[i].already_violated is False


def test_permanently_infeasible_always_violated() -> None:
    sequences = generate_controlled_dataset(
        seed=31,
        sequences_per_scenario=2,
        steps_per_sequence=5,
        scenarios=["permanently_infeasible"],
    )
    for seq in sequences:
        i = seq.known_binding_index
        for case in seq.cases:
            pressure = compute_constraint_pressure(
                case.xB, seq.A, seq.b, seq.models, case.utilities
            )
            assert pressure.feasible is False
            assert pressure.targets[i].already_violated is True


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        generate_controlled_dataset(seed=1, scenarios=["not_a_real_scenario"])


def test_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="steps_per_sequence"):
        generate_controlled_dataset(seed=1, steps_per_sequence=0)


def test_single_step_ramp_does_not_crash() -> None:
    sequences = generate_controlled_dataset(
        seed=5,
        sequences_per_scenario=1,
        steps_per_sequence=1,
        scenarios=["single_constraint_ramp_recoverable"],
    )
    assert len(sequences[0].cases) == 1


def test_generate_one_rejects_unknown_scenario_directly() -> None:
    """generate_controlled_dataset validates before dispatch, but _generate_one
    itself must also refuse an unknown scenario if ever called directly."""
    import numpy as np

    from compitum.constraint_oracle.dataset import _generate_one

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unknown scenario"):
        _generate_one(rng, "not_a_real_scenario", 0, 4)
