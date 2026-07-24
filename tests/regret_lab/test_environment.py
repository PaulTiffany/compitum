"""Controlled dynamic-resource dataset generator -- determinism, grid
quantization, and per-scenario structural sanity checks."""

from __future__ import annotations

from compitum.regret_lab.environment import (
    GRID_UNIT,
    MODEL_NAMES,
    RESOURCE_NAMES,
    SCENARIOS,
    generate_dynamic_dataset,
)


def test_generates_expected_sequence_count_and_ids() -> None:
    seqs = generate_dynamic_dataset(seed=1, sequences_per_scenario=3, steps_per_sequence=6)
    assert len(seqs) == len(SCENARIOS) * 3
    ids = {s.sequence_id for s in seqs}
    assert len(ids) == len(seqs)  # all unique
    for scenario in SCENARIOS:
        assert f"{scenario}-000" in ids
        assert f"{scenario}-002" in ids


def test_deterministic_given_same_seed() -> None:
    a = generate_dynamic_dataset(seed=42, sequences_per_scenario=2, steps_per_sequence=5)
    b = generate_dynamic_dataset(seed=42, sequences_per_scenario=2, steps_per_sequence=5)
    for sa, sb in zip(a, b):
        assert sa.initial_budget == sb.initial_budget
        for ca, cb in zip(sa.cases, sb.cases):
            assert ca.to_dict() == cb.to_dict()


def test_different_seeds_produce_different_utility_jitter() -> None:
    a = generate_dynamic_dataset(seed=1, sequences_per_scenario=1, steps_per_sequence=5)
    b = generate_dynamic_dataset(seed=2, sequences_per_scenario=1, steps_per_sequence=5)
    assert a[0].cases[0].base_utility != b[0].cases[0].base_utility


def test_all_values_are_on_the_grid() -> None:
    seqs = generate_dynamic_dataset(seed=7, sequences_per_scenario=1, steps_per_sequence=8)
    scale = round(1.0 / GRID_UNIT)
    for seq in seqs:
        for r in seq.resource_names:
            assert round(seq.initial_budget[r] * scale) == seq.initial_budget[r] * scale
        for case in seq.cases:
            for m in seq.model_names:
                for r in seq.resource_names:
                    v = case.expected_consumption[m][r]
                    assert abs(round(v * scale) - v * scale) < 1e-6
                    v2 = case.realized_consumption[m][r]
                    assert abs(round(v2 * scale) - v2 * scale) < 1e-6
            for r in seq.resource_names:
                v3 = case.replenishment[r]
                assert abs(round(v3 * scale) - v3 * scale) < 1e-6


def test_model_and_resource_names_declared() -> None:
    assert MODEL_NAMES == ("economy", "standard", "premium")
    assert RESOURCE_NAMES == ("budget", "quota")


def test_permanently_slack_never_binds_at_baseline_rates() -> None:
    seqs = generate_dynamic_dataset(
        seed=3, sequences_per_scenario=2, steps_per_sequence=10, scenarios=("permanently_slack",)
    )
    for seq in seqs:
        remaining = dict(seq.initial_budget)
        for case in seq.cases:
            # Always affording 'premium' (the highest baseline consumer) at
            # every step should never be infeasible in this scenario.
            for r in seq.resource_names:
                assert remaining[r] - case.realized_consumption["premium"][r] >= 0
                remaining[r] -= case.realized_consumption["premium"][r]
                remaining[r] += case.replenishment[r]


def test_single_resource_scarce_period_depletes_quota_mid_sequence() -> None:
    seqs = generate_dynamic_dataset(
        seed=5,
        sequences_per_scenario=1,
        steps_per_sequence=9,
        scenarios=("single_resource_scarce_period",),
    )
    seq = seqs[0]
    remaining_quota = seq.initial_budget["quota"]
    depleted_during_window = False
    window = range(9 // 3, 2 * 9 // 3)
    for t, case in enumerate(seq.cases):
        # Simulate always picking premium greedily with no pricing awareness.
        cost = case.realized_consumption["premium"]["quota"]
        if remaining_quota - cost < 0 and t in window:
            depleted_during_window = True
        remaining_quota = max(0.0, remaining_quota - cost) + case.replenishment["quota"]
    assert depleted_during_window


def test_conserve_enables_better_future_has_large_final_payoff() -> None:
    seqs = generate_dynamic_dataset(
        seed=9,
        sequences_per_scenario=1,
        steps_per_sequence=8,
        scenarios=("conserve_enables_better_future",),
    )
    seq = seqs[0]
    last = seq.cases[-1]
    earlier_max = max(case.base_utility["premium"] for case in seq.cases[:-1])
    assert last.base_utility["premium"] > earlier_max + 5.0
    assert seq.cases[0].replenishment["budget"] == 0.0  # never replenishes


def test_demand_burst_has_one_spike_step() -> None:
    seqs = generate_dynamic_dataset(
        seed=11, sequences_per_scenario=1, steps_per_sequence=8, scenarios=("demand_burst",)
    )
    seq = seqs[0]
    utilities = [case.base_utility["premium"] for case in seq.cases]
    assert max(utilities) - min(utilities) > 10.0


def test_forecast_error_biases_premium_budget_upward() -> None:
    seqs = generate_dynamic_dataset(
        seed=13, sequences_per_scenario=1, steps_per_sequence=8, scenarios=("forecast_error",)
    )
    seq = seqs[0]
    for case in seq.cases:
        assert (
            case.realized_consumption["premium"]["budget"]
            > case.expected_consumption["premium"]["budget"]
        )


def test_delayed_realization_has_nonzero_revelation_delay() -> None:
    seqs = generate_dynamic_dataset(
        seed=17, sequences_per_scenario=1, steps_per_sequence=6, scenarios=("delayed_realization",)
    )
    seq = seqs[0]
    assert all(case.revelation_delay == 2 for case in seq.cases)


def test_premature_conservation_regret_has_one_off_spike() -> None:
    seqs = generate_dynamic_dataset(
        seed=19,
        sequences_per_scenario=1,
        steps_per_sequence=6,
        scenarios=("premature_conservation_regret",),
    )
    seq = seqs[0]
    spike_case = seq.cases[1]
    other_case = seq.cases[0]
    assert spike_case.realized_consumption["standard"]["budget"] > (
        other_case.realized_consumption["standard"]["budget"] * 2.0
    )


def test_multi_resource_interaction_switches_tight_resource_halfway() -> None:
    seqs = generate_dynamic_dataset(
        seed=23,
        sequences_per_scenario=1,
        steps_per_sequence=8,
        scenarios=("multi_resource_interaction",),
    )
    seq = seqs[0]
    first_half_replen = seq.cases[0].replenishment
    second_half_replen = seq.cases[-1].replenishment
    assert first_half_replen["quota"] < first_half_replen["budget"]
    assert second_half_replen["budget"] < second_half_replen["quota"]


def test_case_to_dict_round_trips_all_fields() -> None:
    seqs = generate_dynamic_dataset(seed=1, sequences_per_scenario=1, steps_per_sequence=3)
    case = seqs[0].cases[0]
    d = case.to_dict()
    assert d["step"] == 0
    assert set(d["base_utility"]) == set(MODEL_NAMES)
    assert set(d["expected_consumption"]) == set(MODEL_NAMES)
    assert set(d["realized_consumption"]) == set(MODEL_NAMES)
    assert set(d["replenishment"]) == set(RESOURCE_NAMES)
