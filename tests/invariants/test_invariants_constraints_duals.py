import numpy as np

from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model
from compitum.capabilities import Capabilities


def test_shadow_prices_zero_when_constraints_slack():
    # xB far inside feasible region
    A = np.eye(3)
    b = np.array([10.0, 10.0, 10.0])
    xB = np.array([0.0, 0.0, 0.0])
    solver = ReflectiveConstraintSolver(A, b)

    caps = Capabilities(regions={"US", "EU", "CA"}, tools_allowed={"none"})
    m1 = Model("m1", center=np.zeros(1), capabilities=caps, cost=0.1)
    m2 = Model("m2", center=np.zeros(1), capabilities=caps, cost=0.2)
    models = [m1, m2]
    utilities = {"m1": 1.0, "m2": 0.9}

    _, info = solver.select(xB, models, utilities, context={"region": "US"})
    assert info["feasible"] is True
    # All lambdas zero under clearly slack constraints
    assert all(abs(v) < 1e-12 for v in info["shadow_prices"].values())


def test_shadow_prices_nonnegative_at_boundary():
    A = np.array([[1.0]])
    b = np.array([0.5])
    xB = np.array([0.5])  # at boundary, but feasible
    solver = ReflectiveConstraintSolver(A, b)

    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    m1 = Model("m1", center=np.zeros(1), capabilities=caps, cost=0.1)
    m2 = Model("m2", center=np.zeros(1), capabilities=caps, cost=0.2)
    models = [m1, m2]
    utilities = {"m1": 0.5, "m2": 0.6}  # competitor better but both feasible

    # The solver will pick m2; shadow prices should be ≥ 0
    _, info = solver.select(xB, models, utilities, context={"region": "US"})
    assert info["feasible"] is True
    assert all(v >= 0.0 for v in info["shadow_prices"].values())

