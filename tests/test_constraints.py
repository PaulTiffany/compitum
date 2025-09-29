from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np

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
