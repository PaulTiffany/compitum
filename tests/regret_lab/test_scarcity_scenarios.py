"""Parameterized scarcity/opportunity-cost scenario generator -- structural
sanity checks and determinism, cross-validated where feasible."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.hindsight import compute_hindsight_optimum
from compitum.regret_lab.scarcity_scenarios import (
    CONSUMPTION_ASYMMETRY_LEVELS,
    FORECAST_ERROR_MODES,
    OPPORTUNITY_COST,
    OPPORTUNITY_PREVALENCE_LEVELS,
    PRIMARY_REPLENISHMENT_MODES,
    SCARCITY_MODEL_NAMES,
    SCARCITY_RESOURCE_NAMES,
    ScarcityParams,
    _secondary_windows,
    build_scarcity_sequence,
    generate_corrected_slack_dataset,
    generate_primary_dataset,
    generate_secondary_dataset,
    primary_grid,
    secondary_sweeps,
)


def test_primary_grid_size_matches_declared_axes() -> None:
    grid = primary_grid()
    assert len(grid) == 4 * 3 * 2 * 3  # payoff x tightness x replenishment x timing
    assert len({p.cell_id() for p in grid}) == len(grid)  # all distinct


def test_generate_primary_dataset_size_and_uniqueness() -> None:
    seqs = generate_primary_dataset(seed=1)
    assert len(seqs) == len(primary_grid()) * 3
    assert len({s.sequence_id for s in seqs}) == len(seqs)


def test_generate_primary_dataset_deterministic_across_calls() -> None:
    a = generate_primary_dataset(seed=7)
    b = generate_primary_dataset(seed=7)
    for sa, sb in zip(a, b):
        assert sa.initial_budget == sb.initial_budget
        for ca, cb in zip(sa.cases, sb.cases):
            assert ca.to_dict() == cb.to_dict()


def test_secondary_sweeps_cover_declared_axes() -> None:
    sweeps = secondary_sweeps()
    assert set(sweeps) == {
        "consumption_asymmetry",
        "forecast_error_mode",
        "opportunity_prevalence",
        "replenishment_mode",
    }
    assert len(sweeps["consumption_asymmetry"]) == len(CONSUMPTION_ASYMMETRY_LEVELS)
    assert len(sweeps["forecast_error_mode"]) == len(FORECAST_ERROR_MODES)
    assert len(sweeps["opportunity_prevalence"]) == len(OPPORTUNITY_PREVALENCE_LEVELS)


def test_generate_secondary_dataset_size_per_axis() -> None:
    datasets = generate_secondary_dataset(seed=3)
    assert len(datasets["consumption_asymmetry"]) == len(CONSUMPTION_ASYMMETRY_LEVELS) * 5
    assert len(datasets["forecast_error_mode"]) == len(FORECAST_ERROR_MODES) * 5


def _params(**overrides):
    defaults = dict(
        payoff_ratio=3.0, budget_tightness=1.1, replenishment_mode="none", timing="final"
    )
    defaults.update(overrides)
    return ScarcityParams(**defaults)


def test_slack_budget_makes_opportunity_and_spend_always_affordable() -> None:
    params = _params(budget_tightness=2.0, timing="final")
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, params, steps=12, sequence_id="s")
    # Spending every step and still affording the opportunity at the end
    # must be possible under a genuinely slack budget.
    remaining = seq.initial_budget["budget"]
    for t, case in enumerate(seq.cases):
        if t == 11:
            cost = case.realized_consumption["opportunity"]["budget"]
        else:
            cost = case.realized_consumption["spend"]["budget"]
        assert remaining - cost >= 0
        remaining -= cost
        remaining += case.replenishment["budget"]


def test_severe_budget_requires_conservation_to_reach_opportunity() -> None:
    params = _params(budget_tightness=1.0, timing="final", replenishment_mode="none")
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, params, steps=12, sequence_id="s")
    # Spending every step (not conserving) must NOT leave enough for the
    # opportunity at severe tightness -- otherwise "severe" isn't tight.
    remaining = seq.initial_budget["budget"]
    for t, case in enumerate(seq.cases[:-1]):
        remaining -= case.realized_consumption["spend"]["budget"]
        remaining += case.replenishment["budget"]
    final_cost = seq.cases[-1].realized_consumption["opportunity"]["budget"]
    assert remaining - final_cost < 0


def test_timing_near_mid_final_place_the_window_correctly() -> None:
    rng = np.random.default_rng(0)
    near = build_scarcity_sequence(rng, _params(timing="near"), steps=12, sequence_id="s")
    mid = build_scarcity_sequence(rng, _params(timing="mid"), steps=12, sequence_id="s")
    final = build_scarcity_sequence(rng, _params(timing="final"), steps=12, sequence_id="s")
    assert near.cases[1].base_utility["opportunity"] > 0
    assert mid.cases[6].base_utility["opportunity"] > 0
    assert final.cases[11].base_utility["opportunity"] > 0
    # No window elsewhere for the 'near' sequence.
    assert all(c.base_utility["opportunity"] == 0.0 for i, c in enumerate(near.cases) if i != 1)


def test_payoff_ratio_one_is_a_no_bonus_control() -> None:
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, _params(payoff_ratio=1.0), steps=12, sequence_id="s")
    opportunity_utility = seq.cases[11].base_utility["opportunity"]
    spend_utility = seq.cases[11].base_utility["spend"]
    assert opportunity_utility == pytest.approx(spend_utility)


def test_replenishment_modes_produce_distinct_schedules() -> None:
    rng = np.random.default_rng(0)
    none_seq = build_scarcity_sequence(rng, _params(replenishment_mode="none"), 12, "s")
    partial_seq = build_scarcity_sequence(rng, _params(replenishment_mode="partial"), 12, "s")
    periodic_seq = build_scarcity_sequence(rng, _params(replenishment_mode="periodic"), 12, "s")
    delayed_seq = build_scarcity_sequence(rng, _params(replenishment_mode="delayed"), 12, "s")

    assert all(c.replenishment["budget"] == 0.0 for c in none_seq.cases)
    assert all(c.replenishment["budget"] == 0.5 for c in partial_seq.cases)
    assert sum(c.replenishment["budget"] for c in periodic_seq.cases) == pytest.approx(
        sum(c.replenishment["budget"] for c in partial_seq.cases), abs=0.5
    )
    assert all(c.replenishment["budget"] == 0.0 for c in delayed_seq.cases[:6])
    assert all(c.replenishment["budget"] == 1.0 for c in delayed_seq.cases[6:])


def test_consumption_asymmetry_scales_spend_cost_only() -> None:
    rng = np.random.default_rng(0)
    mild = build_scarcity_sequence(rng, _params(consumption_asymmetry=1.2), 12, "s")
    strong = build_scarcity_sequence(rng, _params(consumption_asymmetry=4.0), 12, "s")
    # 1.2 is not itself on the 0.25 grid used everywhere for hindsight-DP
    # exactness, so it is quantized to the nearest grid value (1.25).
    assert mild.cases[0].realized_consumption["spend"]["budget"] == pytest.approx(1.25)
    assert strong.cases[0].realized_consumption["spend"]["budget"] == pytest.approx(4.0)
    assert mild.cases[0].realized_consumption["conserve"]["budget"] == pytest.approx(1.0)
    assert strong.cases[0].realized_consumption["conserve"]["budget"] == pytest.approx(1.0)


def test_forecast_error_over_understates_affordability() -> None:
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, _params(forecast_error_mode="over"), 12, "s")
    case = seq.cases[11]
    assert (
        case.expected_consumption["opportunity"]["budget"]
        > case.realized_consumption["opportunity"]["budget"]
    )


def test_forecast_error_under_overstates_affordability() -> None:
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, _params(forecast_error_mode="under"), 12, "s")
    case = seq.cases[11]
    assert (
        case.expected_consumption["opportunity"]["budget"]
        < case.realized_consumption["opportunity"]["budget"]
    )


def test_forecast_error_delayed_sets_revelation_delay_on_the_window_step() -> None:
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, _params(forecast_error_mode="delayed"), 12, "s")
    assert seq.cases[11].revelation_delay == 3
    assert seq.cases[0].revelation_delay == 0


def test_opportunity_prevalence_moderate_adds_secondary_windows() -> None:
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, _params(opportunity_prevalence="moderate"), 12, "s")
    window_steps = [i for i, c in enumerate(seq.cases) if c.base_utility["opportunity"] > 0]
    assert len(window_steps) >= 2  # primary (final) plus at least one secondary


def test_secondary_windows_skips_a_candidate_that_coincides_with_t_opp() -> None:
    # steps//4 == 3 for steps=12; when t_opp is also 3, that candidate must
    # be skipped (only the other candidate, 9, becomes a window).
    windows = _secondary_windows(
        np.random.default_rng(0), _params(opportunity_prevalence="moderate"), steps=12, t_opp=3
    )
    assert 3 not in windows
    assert 9 in windows


def test_opportunity_prevalence_rare_has_exactly_one_window() -> None:
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, _params(opportunity_prevalence="rare"), 12, "s")
    window_steps = [i for i, c in enumerate(seq.cases) if c.base_utility["opportunity"] > 0]
    assert window_steps == [11]


def test_opportunity_prevalence_stochastic_is_seed_dependent() -> None:
    seq_a = build_scarcity_sequence(
        np.random.default_rng(1), _params(opportunity_prevalence="stochastic"), 12, "s"
    )
    seq_b = build_scarcity_sequence(
        np.random.default_rng(2), _params(opportunity_prevalence="stochastic"), 12, "s"
    )
    windows_a = [i for i, c in enumerate(seq_a.cases) if c.base_utility["opportunity"] > 0]
    windows_b = [i for i, c in enumerate(seq_b.cases) if c.base_utility["opportunity"] > 0]
    assert windows_a != windows_b or windows_a == [11]  # extremely unlikely to tie by chance


def test_opportunity_always_declared_but_infeasible_off_window() -> None:
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, _params(timing="final"), 12, "s")
    assert seq.cases[0].realized_consumption["opportunity"]["budget"] == OPPORTUNITY_COST or (
        seq.cases[0].realized_consumption["opportunity"]["budget"] > seq.initial_budget["budget"]
    )


def test_model_and_resource_names_are_fixed() -> None:
    assert SCARCITY_MODEL_NAMES == ("conserve", "spend", "opportunity")
    assert SCARCITY_RESOURCE_NAMES == ("budget",)


def test_hindsight_oracle_runs_cleanly_on_a_generated_sequence() -> None:
    rng = np.random.default_rng(0)
    seq = build_scarcity_sequence(rng, _params(budget_tightness=1.1, timing="final"), 12, "s")
    result = compute_hindsight_optimum(seq)
    assert result.exact is True
    assert result.value >= 0.0


def test_cell_id_reflects_all_seven_axes() -> None:
    params = ScarcityParams(
        payoff_ratio=3.0,
        budget_tightness=1.1,
        replenishment_mode="none",
        timing="final",
        consumption_asymmetry=2.0,
        forecast_error_mode="none",
        opportunity_prevalence="rare",
    )
    cell_id = params.cell_id()
    for fragment in ("pr3.0", "bt1.1", "repnone", "tfinal", "ca2.0", "fenone", "oprare"):
        assert fragment in cell_id


def test_tightness_reference_rate_changes_initial_budget() -> None:
    rng = np.random.default_rng(0)
    default_ref = build_scarcity_sequence(rng, _params(timing="final"), 12, "s")
    natural_ref = build_scarcity_sequence(
        rng, _params(timing="final"), 12, "s", tightness_reference_rate=2.0
    )
    # consumption_asymmetry default is 2.0, double CONSERVE_RATE (1.0), so
    # the natural-reference budget must be larger for the same cell.
    assert natural_ref.initial_budget["budget"] > default_ref.initial_budget["budget"]


def test_generate_corrected_slack_dataset_covers_only_near_and_mid_timing() -> None:
    seqs = generate_corrected_slack_dataset(seed=5)
    grid = [p for p in primary_grid() if p.timing in ("near", "mid")]
    assert len(seqs) == len(grid) * 3  # 3 seeds per cell
    assert all(seq.sequence_id.startswith("corrected-") for seq in seqs)
    assert all("tfinal" not in s.scenario for s in seqs)


def test_generate_corrected_slack_dataset_is_calibrated_against_natural_rate() -> None:
    seqs = generate_corrected_slack_dataset(seed=5)
    slack_near = next(
        s for s in seqs if "bt2.0" in s.scenario and "tnear" in s.scenario and "ca2.0" in s.scenario
    )
    # Corrected reference uses consumption_asymmetry (2.0 default) instead
    # of CONSERVE_RATE (1.0): budget_tightness=2.0 at t_opp=1 now yields
    # 2.0 * (1 * 2.0 + 5.0) = 14.0, not the original 2.0 * (1*1.0+5.0)=12.0.
    assert slack_near.initial_budget["budget"] == pytest.approx(14.0)


def test_generate_corrected_slack_dataset_deterministic() -> None:
    a = generate_corrected_slack_dataset(seed=9)
    b = generate_corrected_slack_dataset(seed=9)
    for sa, sb in zip(a, b):
        assert sa.initial_budget == sb.initial_budget


def test_unknown_timing_raises() -> None:
    with pytest.raises(ValueError, match="timing"):
        build_scarcity_sequence(np.random.default_rng(0), _params(timing="never"), 12, "s")


def test_unknown_replenishment_mode_raises() -> None:
    with pytest.raises(ValueError, match="replenishment_mode"):
        build_scarcity_sequence(
            np.random.default_rng(0), _params(replenishment_mode="magic"), 12, "s"
        )
