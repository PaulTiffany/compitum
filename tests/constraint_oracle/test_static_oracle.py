"""Hand-constructed cases with known answers for the exact constraint-pressure
oracle, cross-checked against a numeric bisection search of the actual frozen
``ReflectiveConstraintSolver``.

Structural invariant exercised throughout (see static.py's module docstring):
a single violated constraint makes EVERY model simultaneously linearly
infeasible, so "a feasible route already exists" and "some constraint is
currently violated" never co-occur. There are exactly two live branches:
linear-feasible (capability alone decides the pick) and linear-infeasible
(unconditionally the frozen ``infeasible_fallback`` path).
"""

from __future__ import annotations

import numpy as np
import pytest

from compitum.capabilities import Capabilities
from compitum.constraint_oracle.static import compute_constraint_pressure
from compitum.constraint_oracle.validation import numeric_critical_relaxation
from compitum.models import Model

# Diagonal A (identity): row i of A@xB is exactly xB[i], matching the
# project's own default constraints_us_default.yaml (4 diagonal rows).
A = np.eye(4)
B = np.array([2.0, 2.0, 0.0, 0.0])


def _model(name: str, regions: set) -> Model:
    return Model(
        name=name,
        center=np.zeros(2),
        capabilities=Capabilities(regions=regions, tools_allowed={"none"}),
        cost=0.1,
    )


ALL_REGIONS = {"US", "CA", "EU"}


def test_fully_feasible_no_suppressed_competitor() -> None:
    """All constraints slack; the optimal-utility model is already chosen.
    Nothing is violated, so every target is simply not-currently-binding."""
    models = [_model("winner", ALL_REGIONS), _model("loser", ALL_REGIONS)]
    utilities = {"winner": 1.0, "loser": 0.5}
    xB = np.array([0.0, 0.0, 0.0, 0.0])  # deep inside b=[2,2,0,0]

    result = compute_constraint_pressure(xB, A, B, models, utilities)

    assert result.feasible is True
    assert result.selected_model == "winner"
    assert result.capability_blocked_high_utility_models == []
    for target in result.targets:
        assert target.reason == "not_currently_binding"
        assert target.critical_relaxation is None
        assert not target.already_violated


def test_capability_blocked_high_utility_model_when_otherwise_feasible() -> None:
    """The globally-optimal model is blocked purely by region capability, but
    nothing is currently violated -- constraint relaxation is simply
    irrelevant here (correctly "not_currently_binding"), and the case-level
    diagnostic records the capability block separately."""
    models = [_model("blocked_best", {"EU"}), _model("chosen", ALL_REGIONS)]
    utilities = {"blocked_best": 5.0, "chosen": 1.0}
    xB = np.array([0.0, 0.0, 0.0, 0.0])
    context = {"region": "US"}

    result = compute_constraint_pressure(xB, A, B, models, utilities, context=context)

    assert result.feasible is True
    assert result.selected_model == "chosen"
    assert result.capability_blocked_high_utility_models == ["blocked_best"]
    for target in result.targets:
        assert target.reason == "not_currently_binding"
        assert target.critical_relaxation is None


def test_sole_violated_constraint_recovers_feasibility_with_winner_change() -> None:
    """Constraint 0 alone is violated. The frozen infeasible_fallback path
    would pick the highest-utility model regardless of its capability;
    relaxing the sole violated constraint recovers a real, capability-gated
    pick, which can differ from the fallback's arbitrary choice."""
    models = [_model("fallback_pick", {"EU"}), _model("real_winner", ALL_REGIONS)]
    utilities = {"fallback_pick": 9.0, "real_winner": 5.0}
    context = {"region": "US"}
    # xB[0] = 2.5 -> row0 = 2.5 > b[0]=2.0 -> violated by exactly 0.5
    xB = np.array([2.5, 0.0, 0.0, 0.0])

    result = compute_constraint_pressure(xB, A, B, models, utilities, context=context)

    assert result.feasible is False
    assert result.fallback_only is True
    assert result.selected_model == "fallback_pick"  # highest utility, regardless of capability

    t0 = result.targets[0]
    assert t0.already_violated is True
    assert t0.reason == "recovers_feasibility"
    assert t0.recovers_feasibility is True
    assert t0.critical_relaxation == pytest.approx(0.5 - 1e-10, abs=1e-9)
    assert t0.best_suppressed_competitor == "real_winner"  # only capable model
    assert t0.discontinuous_winner_change is False

    for i in (1, 2, 3):
        assert result.targets[i].reason == "not_currently_binding"
        assert result.targets[i].critical_relaxation is None

    # Cross-check with the real frozen solver: the winner genuinely changes
    # from "fallback_pick" to "real_winner" once constraint 0 is relaxed.
    numeric = numeric_critical_relaxation(0, xB, A, B, models, utilities, context=context)
    assert numeric is not None
    assert numeric == pytest.approx(t0.critical_relaxation, abs=1e-6)


def test_multi_constraint_block_reports_blocked_by_other() -> None:
    """Constraints 0 and 1 are both violated; relaxing either alone cannot
    fix the case (the shared np.all(...) check still fails on the other), so
    both targets must report the other as blocking, and the real solver's
    winner must never change no matter how far constraint 0 alone relaxes."""
    models = [_model("suppressed", ALL_REGIONS), _model("current", ALL_REGIONS)]
    utilities = {"suppressed": 9.0, "current": 1.0}
    xB = np.array([2.5, 2.5, 0.0, 0.0])

    result = compute_constraint_pressure(xB, A, B, models, utilities)

    assert result.feasible is False
    assert result.fallback_only is True
    t0, t1 = result.targets[0], result.targets[1]
    assert t0.reason == "blocked_by_other_constraint"
    assert t0.blocking_constraint_indices == [1]
    assert t1.reason == "blocked_by_other_constraint"
    assert t1.blocking_constraint_indices == [0]
    assert t0.critical_relaxation is None
    assert t1.critical_relaxation is None
    assert result.jointly_blocking_constraint_indices == [0, 1]

    numeric = numeric_critical_relaxation(0, xB, A, B, models, utilities, upper_bound=1000.0)
    assert numeric is None


def test_discontinuous_winner_change_when_two_competitors_tie() -> None:
    """Sole violated constraint; two capable competitors share the exact
    same top utility -- both become feasible simultaneously (shared slack),
    a genuine tie in who the recovered pick would be."""
    models = [
        _model("tie_a", ALL_REGIONS),
        _model("tie_b", ALL_REGIONS),
        _model("current", ALL_REGIONS),
    ]
    utilities = {"tie_a": 9.0, "tie_b": 9.0, "current": 1.0}
    xB = np.array([2.5, 0.0, 0.0, 0.0])

    result = compute_constraint_pressure(xB, A, B, models, utilities)

    assert result.feasible is False
    t0 = result.targets[0]
    assert t0.reason == "recovers_feasibility"
    assert t0.discontinuous_winner_change is True
    assert set(t0.tied_competitors) == {"tie_a", "tie_b"}


def test_globally_infeasible_all_capability_blocked_but_nothing_violated() -> None:
    """Case is fully linear-feasible (nothing violated) but every model
    fails capabilities -- still the frozen infeasible_fallback branch at the
    case level, yet every per-constraint target is "not_currently_binding"
    since no constraint relaxation is even relevant to a capability-only
    block."""
    models = [_model("only", {"EU"})]
    utilities = {"only": 1.0}
    xB = np.array([0.0, 0.0, 0.0, 0.0])
    context = {"region": "US"}

    result = compute_constraint_pressure(xB, A, B, models, utilities, context=context)

    assert result.feasible is False
    assert result.fallback_only is True
    assert result.selected_model == "only"
    for target in result.targets:
        assert target.reason == "not_currently_binding"
        assert target.critical_relaxation is None


def test_sole_violated_constraint_with_zero_capable_models() -> None:
    """Sole violated constraint, but even the only model in the pool fails
    capabilities: relaxing this constraint can never produce a feasible
    pick, distinct from "blocked_by_other_constraint"."""
    models = [_model("only", {"EU"})]
    utilities = {"only": 1.0}
    context = {"region": "US"}
    xB = np.array([2.5, 0.0, 0.0, 0.0])  # constraint 0 alone violated

    result = compute_constraint_pressure(xB, A, B, models, utilities, context=context)

    assert result.feasible is False
    t0 = result.targets[0]
    assert t0.reason == "capability_blocked_only"
    assert t0.critical_relaxation is None
    for i in (1, 2, 3):
        assert result.targets[i].reason == "not_currently_binding"


def test_globally_infeasible_single_blocking_constraint_recovers() -> None:
    """Linear-infeasible on constraint 2 alone; relaxing it recovers
    feasibility for the highest-utility capability-eligible model. No
    utility baseline exists in this branch, so marginal_utility_improvement
    is deliberately left unset."""
    models = [_model("m1", ALL_REGIONS), _model("m2", ALL_REGIONS)]
    utilities = {"m1": 3.0, "m2": 7.0}
    # constraint index 2 (pii_level <= 0): xB[2]=0.2 > b[2]=0.0 -> violated
    xB = np.array([0.0, 0.0, 0.2, 0.0])

    result = compute_constraint_pressure(xB, A, B, models, utilities)

    assert result.feasible is False
    assert result.fallback_only is True
    t2 = result.targets[2]
    assert t2.reason == "recovers_feasibility"
    assert t2.recovers_feasibility is True
    assert t2.critical_relaxation == pytest.approx(0.2 - 1e-10, abs=1e-9)
    assert t2.best_suppressed_competitor == "m2"  # higher utility of the two
    assert t2.marginal_utility_improvement is None

    for i in (0, 1, 3):
        assert result.targets[i].reason == "not_currently_binding"


def test_near_binding_flag() -> None:
    models = [_model("only", ALL_REGIONS)]
    utilities = {"only": 1.0}
    xB = np.array([1.9995, 0.0, 0.0, 0.0])  # slack = 0.0005, within default 1e-3
    result = compute_constraint_pressure(xB, A, B, models, utilities)
    assert result.targets[0].near_binding is True
    assert result.targets[0].already_violated is False


def test_to_dict_is_json_shaped_and_names_no_shadow_price() -> None:
    models = [_model("only", ALL_REGIONS)]
    utilities = {"only": 1.0}
    xB = np.array([0.0, 0.0, 0.0, 0.0])
    result = compute_constraint_pressure(xB, A, B, models, utilities)
    payload = result.to_dict()
    assert payload["schema"] == "compitum.constraint-pressure-oracle/v1"
    assert isinstance(payload["targets"], list)
    assert "shadow_price" not in str(payload).lower()
    assert "dual" not in str(payload).lower()


def test_numeric_critical_relaxation_respects_max_iterations() -> None:
    """Exercise the loop exhausting max_iterations without ever hitting the
    tolerance break, still returning a sensible (if less precise) bound."""
    models = [_model("fallback_pick", {"EU"}), _model("real_winner", ALL_REGIONS)]
    utilities = {"fallback_pick": 9.0, "real_winner": 5.0}
    context = {"region": "US"}
    xB = np.array([2.5, 0.0, 0.0, 0.0])

    numeric = numeric_critical_relaxation(
        0, xB, A, B, models, utilities, context=context, max_iterations=1
    )
    assert numeric is not None
    assert 0.0 < numeric <= 100.0
