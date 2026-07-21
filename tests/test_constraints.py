from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from compitum.capabilities import Capabilities
from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model


def test_solver_basic_feasible() -> None:
    """Tests the basic case where all models are feasible."""
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    pgd = np.array([0.5])
    caps = Capabilities(set(), set())
    models = [
        Model(name="a", center=np.array([]), capabilities=caps, cost=0.0),
        Model(name="b", center=np.array([]), capabilities=caps, cost=0.0),
    ]
    utilities = {"a": 0.2, "b": 0.3}
    m_star, info = solver.select(pgd, models, utilities)
    assert m_star.name == "b"
    assert info["feasible"] is True
    # "status" and "violations" were never checked on the feasible path --
    # only "feasible" itself -- so mutations to those dict keys/values
    # would survive.
    assert info["status"] == "optimal"
    assert info["violations"] == []


def test_solver_no_viable_models() -> None:
    """Tests the case where no models are feasible due to constraints."""
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    pgd_infeasible = np.array([2.0])  # Violates constraint
    caps = Capabilities(set(), set())
    models = [
        Model(name="a", center=np.array([]), capabilities=caps, cost=0.0),
        Model(name="b", center=np.array([]), capabilities=caps, cost=0.0),
    ]
    utilities = {"a": 0.2, "b": 0.9}

    m_star, info = solver.select(pgd_infeasible, models, utilities)
    assert info["feasible"] is False
    assert m_star.name == "b"  # Should return model with max utility
    # Same gap as above, on the infeasible-fallback path.
    assert info["status"] == "infeasible_fallback"
    assert info["violations"] == ["all_models_violate_constraints"]


def test_solver_feasibility_at_exact_boundary() -> None:
    """No existing test uses x exactly at A@x == b -- the <= vs < choice in
    `np.all(self.A @ xB <= self.b + 1e-10)` was never exercised at the actual
    boundary, only clearly inside/outside it."""
    caps = Capabilities(set(), set())
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    models = [Model(name="a", center=np.array([]), capabilities=caps, cost=0.0)]
    _, info = solver.select(np.array([1.0]), models, {"a": 0.5})
    assert info["feasible"] is True


def test_solver_feasibility_at_exact_epsilon_boundary() -> None:
    """The test above uses x == b exactly, where `A@x <= b + 1e-10` and
    `A@x < b + 1e-10` both agree (True either way, since b+1e-10 > b) -- the
    `1e-10` tolerance itself was never actually put at stake. Construct x so
    `A@x` lands exactly on `b + 1e-10` (same literal arithmetic as the
    source), where `<=` (True, feasible) and `<` (False, infeasible)
    genuinely disagree."""
    caps = Capabilities(set(), set())
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    models = [Model(name="a", center=np.array([]), capabilities=caps, cost=0.0)]
    _, info = solver.select(np.array([1.0 + 1e-10]), models, {"a": 0.5})
    assert info["feasible"] is True


def test_solver_missing_utility_defaults_to_worst_not_best() -> None:
    """No existing test omits a model from the utilities dict -- the
    `-np.inf` default (vs a mutated `+np.inf`) that makes an unscored model
    lose, not win, was never exercised."""
    caps = Capabilities(set(), set())
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    models = [
        Model(name="a", center=np.array([]), capabilities=caps, cost=0.0),
        Model(name="b", center=np.array([]), capabilities=caps, cost=0.0),
    ]
    # "b" is intentionally absent from utilities.
    m_star, _ = solver.select(np.array([0.5]), models, {"a": 0.5})
    assert m_star.name == "a"


def test_solver_shadow_price_competitor_missing_utility_defaults_to_worst() -> None:
    """The missing-utility test above only checks m_star selection (which
    exercises the sort key's `-np.inf` default), never the *separate*
    `utilities.get(competitor.name, -np.inf)` inside the shadow-price loop
    (a different line, same default pattern). A missing competitor must
    still lose there too -- shadow_price must stay 0.0, not become a huge
    value from a wrongly-defaulted +inf utility beating the real m_star."""
    caps = Capabilities(set(), set())
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    models = [
        Model(name="m_star", center=np.array([]), capabilities=caps, cost=0.0),
        Model(name="competitor", center=np.array([]), capabilities=caps, cost=0.0),
    ]
    # "competitor" is intentionally absent from utilities.
    _, info = solver.select(np.array([0.5]), models, {"m_star": 0.5})
    assert info["shadow_prices"]["lambda_0"] == 0.0


def test_solver_capability_support_filters_model() -> None:
    """Tests that a model is correctly filtered out by its `supports` method."""
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    pgd = np.array([0.5])  # Feasible from Ax<=b perspective

    caps_a = Capabilities(set(), set())
    caps_b_mock = MagicMock()
    caps_b_mock.supports.return_value = False

    models = [
        Model(name="a", center=np.array([]), capabilities=caps_a, cost=0.0),
        Model(name="b", center=np.array([]), capabilities=caps_b_mock, cost=0.0),
    ]
    utilities = {"a": 0.2, "b": 0.9}

    m_star, info = solver.select(pgd, models, utilities)

    assert m_star.name == "a"
    assert info["feasible"] is True
    caps_b_mock.supports.assert_called_with(pgd)


def test_solver_shadow_price_and_viable_competitor() -> None:
    """Final test to cover all branches in the shadow price calculation."""
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    pgd = np.array([0.5])

    caps_true = Capabilities(set(), set())
    caps_false_mock = MagicMock()
    caps_false_mock.supports.return_value = False

    m_viable_low_util = Model(
        name="viable_low", center=np.array([]), capabilities=caps_true, cost=0.0
    )
    m_viable_m_star = Model(
        name="viable_m_star", center=np.array([]), capabilities=caps_true, cost=0.0
    )
    m_non_viable_high_util = Model(
        name="non_viable_high", center=np.array([]), capabilities=caps_false_mock, cost=0.0
    )

    models = [m_viable_low_util, m_viable_m_star, m_non_viable_high_util]
    utilities = {"viable_low": 0.1, "viable_m_star": 0.5, "non_viable_high": 0.9}

    m_star, info = solver.select(pgd, models, utilities)

    assert m_star.name == "viable_m_star"
    assert info["feasible"] is True
    # The shadow price is 0 because the non-viable model is non-viable due to capabilities
    # and relaxing the b constraint doesn't change that.
    assert info["shadow_prices"]["lambda_0"] == 0


def test_solver_shadow_price_positive_when_capability_becomes_true() -> None:
    """
    Forces ok=True in the shadow-price loop:
    - competitor is *not* in 'viable' because supports() returns False the first time
    - under 'relaxation', supports() returns True
    This drives the if ok: branch (line 35) and covers arc 35->36 and 35->31.
    """

    class FlippingCaps(Capabilities):
        def __init__(self) -> None:
            super().__init__(set(), set())
            self.calls = 0

        def supports(self, pgd_vector: Any, context: Dict[str, Any] | None = None) -> bool:
            self.calls += 1
            # 1st call (filtering): False → model excluded from 'viable'
            # 2nd call (inside shadow-price check): True → ok becomes True
            return self.calls > 1

    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    x = np.array([0.5])  # A·x <= b holds

    good = Capabilities(set(), set())  # viable model (m_star)
    flip = FlippingCaps()  # non-viable at first, then viable

    models = [
        Model(name="best", center=np.array([]), capabilities=good, cost=0.0),
        Model(name="better", center=np.array([]), capabilities=flip, cost=0.0),
    ]
    utilities = {"best": 0.5, "better": 0.9}

    m_star, info = solver.select(x, models, utilities)

    assert m_star.name == "best"
    # Shadow price must now be positive because the "better" competitor
    # becomes viable under the (simulated) relaxation.
    assert info["shadow_prices"]["lambda_0"] > 0.0
    # Existing test above only checks the sign -- assert the exact value too,
    # so a mutated relaxation epsilon (1e-5) or a broken division doesn't
    # survive just because the sign happens to still be positive.
    assert np.isclose(info["shadow_prices"]["lambda_0"], (0.9 - 0.5) / 1e-5)


def test_solver_shadow_price_relaxation_epsilon_sign_is_plus() -> None:
    """`b_relaxed[i] += 1e-5` -> `-= 1e-5` was never exercised where the sign
    actually matters: the shared `xB` makes the *unrelaxed* constraint check
    identical for every model, so a competitor's only path to
    "excluded from viable but viable under relaxation" is via capability,
    not the constraint -- unless the unrelaxed check passes with a margin
    smaller than 1e-5, where tightening b by 1e-5 (the sign-flip bug) can
    flip a passing check to failing. A `FlippingCaps`-style capability
    (False on the first call so the competitor is excluded from `viable`
    despite a higher raw utility than m_star, True afterward) isolates the
    relaxation epsilon itself as the only remaining variable."""

    class OnceFalseCaps(Capabilities):
        def __init__(self) -> None:
            super().__init__(set(), set())
            self.calls = 0

        def supports(self, pgd_vector: Any, context: Dict[str, Any] | None = None) -> bool:
            self.calls += 1
            return self.calls > 1

    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    # Barely inside the *unrelaxed* boundary (b + 1e-10) -- +1e-5 relaxation
    # keeps it comfortably feasible; -1e-5 pushes it back out.
    xB = np.array([1.0 + 0.5e-10])

    m_star = Model(
        name="m_star", center=np.array([]), capabilities=Capabilities(set(), set()), cost=0.0
    )
    competitor = Model(
        name="competitor", center=np.array([]), capabilities=OnceFalseCaps(), cost=0.0
    )

    _, info = solver.select(xB, [m_star, competitor], {"m_star": 0.5, "competitor": 0.9})
    assert np.isclose(info["shadow_prices"]["lambda_0"], (0.9 - 0.5) / 1e-5)


def test_solver_shadow_price_context_passed_through_on_every_call() -> None:
    """`if context is None: ... else: ...supports(xB, context=context)` was
    flipped to `if context is not None:`, swapping which branch runs -- with
    a real non-None `context`, this makes the shadow-price loop's own
    `ok_cap` call drop the `context` kwarg entirely, while `_is_feasible`'s
    *own*, unmutated branching (used both during filtering and inside the
    relaxed-constraint check) still passes it. Checking that *every*
    recorded call included `context` catches the one that silently
    stopped."""
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    xB = np.array([0.5])
    context = {"region": "US"}

    m_star = Model(
        name="m_star", center=np.array([]), capabilities=Capabilities(set(), set()), cost=0.0
    )
    competitor_caps = MagicMock()
    competitor_caps.supports.return_value = True
    competitor = Model(
        name="competitor", center=np.array([]), capabilities=competitor_caps, cost=0.0
    )

    solver.select(xB, [m_star, competitor], {"m_star": 0.5, "competitor": 0.3}, context=context)

    assert competitor_caps.supports.call_args_list  # sanity: it was called at all
    for call in competitor_caps.supports.call_args_list:
        assert call.kwargs.get("context") == context


def test_solver_shadow_price_ok_cap_false_keeps_competitor_non_viable() -> None:
    """`if not ok_cap: is_competitor_viable_relaxed = False` -> `= True` --
    a competitor whose capability genuinely fails must never contribute a
    nonzero shadow price, even if its utility would otherwise beat m_star's.
    A capability mock that returns False exactly twice (the filtering call,
    then the shadow-price loop's own `ok_cap` call) and True afterward (the
    separate internal check inside `_is_feasible`) isolates this specific
    line from the independent `_is_feasible`-based check that follows it."""

    class TwiceFalseCaps(Capabilities):
        def __init__(self) -> None:
            super().__init__(set(), set())
            self.calls = 0

        def supports(self, pgd_vector: Any, context: Dict[str, Any] | None = None) -> bool:
            self.calls += 1
            return self.calls > 2

    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    xB = np.array([0.5])

    m_star = Model(
        name="m_star", center=np.array([]), capabilities=Capabilities(set(), set()), cost=0.0
    )
    competitor = Model(
        name="competitor", center=np.array([]), capabilities=TwiceFalseCaps(), cost=0.0
    )

    _, info = solver.select(xB, [m_star, competitor], {"m_star": 0.5, "competitor": 0.9})
    assert info["shadow_prices"]["lambda_0"] == 0.0


def test_solver_shadow_price_utility_tie_does_not_break_loop_early() -> None:
    """`if utility_competitor > utility_m_star:` -> `>=` -- at an exact tie,
    both comparisons compute the *same* shadow price (`(tie - tie) / 1e-5 ==
    0.0`), so the value alone can't distinguish them. The mutant's `break`
    on a tie, though, would skip every competitor that sorts after the tied
    one -- checked here via a later competitor's mock never getting called."""
    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    xB = np.array([0.5])
    caps = Capabilities(set(), set())

    m_star = Model(name="m_star", center=np.array([]), capabilities=caps, cost=0.0)
    tied_competitor = Model(name="tied", center=np.array([]), capabilities=caps, cost=0.0)
    later_caps = MagicMock()
    later_caps.supports.return_value = True
    later_competitor = Model(name="later", center=np.array([]), capabilities=later_caps, cost=0.0)

    utilities = {"m_star": 0.5, "tied": 0.5, "later": 0.1}
    _, info = solver.select(xB, [m_star, tied_competitor, later_competitor], utilities)

    assert info["shadow_prices"]["lambda_0"] == 0.0
    assert later_caps.supports.called


def test_solver_shadow_price_keeps_first_winner_not_last() -> None:
    """`break` -> `continue` after a competitor beats m_star's utility --
    the mutant would let a *later*, weaker qualifying competitor overwrite
    an already-found (larger) shadow price. Two competitors both beat
    m_star, in descending-utility sorted order, so the first one's shadow
    price must be the one that sticks. Both use the same
    once-False-then-True capability trick as the relaxation test above --
    if they were viable from the start, one of *them* (higher raw utility)
    would become m_star instead of the intended model."""

    class OnceFalseCaps(Capabilities):
        def __init__(self) -> None:
            super().__init__(set(), set())
            self.calls = 0

        def supports(self, pgd_vector: Any, context: Dict[str, Any] | None = None) -> bool:
            self.calls += 1
            return self.calls > 1

    A = np.eye(1)
    b = np.array([1.0])
    solver = ReflectiveConstraintSolver(A, b)
    xB = np.array([0.5])

    m_star = Model(
        name="m_star", center=np.array([]), capabilities=Capabilities(set(), set()), cost=0.0
    )
    strong = Model(name="strong", center=np.array([]), capabilities=OnceFalseCaps(), cost=0.0)
    weak = Model(name="weak", center=np.array([]), capabilities=OnceFalseCaps(), cost=0.0)

    utilities = {"m_star": 0.5, "strong": 0.9, "weak": 0.7}
    m, info = solver.select(xB, [strong, weak, m_star], utilities)

    assert m.name == "m_star"
    assert np.isclose(info["shadow_prices"]["lambda_0"], (0.9 - 0.5) / 1e-5)


def test_solver_infeasible_fallback_zeros_all_constraint_shadow_prices() -> None:
    """No existing infeasible test uses more than one constraint row, so the
    `for i in range(len(self.b))` loop populating every lambda_i with 0.0 in
    the fallback branch was only ever exercised for a single index."""
    A = np.eye(2)
    b = np.array([1.0, 1.0])
    solver = ReflectiveConstraintSolver(A, b)
    pgd_infeasible = np.array([2.0, 2.0])
    caps = Capabilities(set(), set())
    models = [Model(name="a", center=np.array([]), capabilities=caps, cost=0.0)]

    _, info = solver.select(pgd_infeasible, models, {"a": 0.5})
    assert info["feasible"] is False
    assert info["shadow_prices"] == {"lambda_0": 0.0, "lambda_1": 0.0}


@given(
    utilities_list=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    ),
    b_val=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    xB_val=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_solver_select_invariants_hold_across_random_inputs(
    utilities_list: list, b_val: float, xB_val: float
) -> None:
    """Property-based complement to the targeted unit tests above: rather
    than engineering one exact scenario per mutant, sweep `select()` across
    many random utility assignments and a random feasible/infeasible `xB`,
    and check the invariants that must hold regardless of the specific
    numbers -- exercising the same sort/tie/feasibility/shadow-price code
    paths from a different angle. All models share simple always-feasible
    capabilities here, so "viable" reduces to "all models", keeping the
    oracle for "which model should win" simple and independently checkable."""
    A = np.eye(1)
    b = np.array([b_val])
    solver = ReflectiveConstraintSolver(A, b)
    caps = Capabilities(set(), set())
    models = [
        Model(name=f"m{i}", center=np.array([]), capabilities=caps, cost=0.0)
        for i in range(len(utilities_list))
    ]
    utilities = {f"m{i}": u for i, u in enumerate(utilities_list)}
    xB = np.array([xB_val])

    m_star, info = solver.select(xB, models, utilities)

    # The returned model is always one of the inputs.
    assert m_star.name in {m.name for m in models}

    # shadow_prices always has exactly one entry per constraint row.
    assert set(info["shadow_prices"].keys()) == {"lambda_0"}

    # Shadow prices are never negative (they're either 0.0 or a positive
    # utility gap divided by a positive epsilon).
    assert all(v >= 0.0 for v in info["shadow_prices"].values())

    expected_feasible = bool(np.all(A @ xB <= b + 1e-10))
    assert info["feasible"] == expected_feasible

    if info["feasible"]:
        assert info["status"] == "optimal"
        assert info["violations"] == []
        # All models share the same trivial capability and the same xB, so
        # every model is equally "viable" here -- m_star must be whichever
        # one has the maximum utility.
        assert utilities[m_star.name] == max(utilities.values())
    else:
        assert info["status"] == "infeasible_fallback"
        assert info["violations"] == ["all_models_violate_constraints"]
        assert all(v == 0.0 for v in info["shadow_prices"].values())
