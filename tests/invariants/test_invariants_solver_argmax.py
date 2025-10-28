from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model
from compitum.capabilities import Capabilities
import numpy as np


def test_argmax_stability_under_positive_shift():
    # Simple feasibility
    A = np.eye(1)
    b = np.array([10.0])
    xB = np.array([0.0])
    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    m1 = Model("m1", center=np.zeros(1), capabilities=caps, cost=0.1)
    m2 = Model("m2", center=np.zeros(1), capabilities=caps, cost=0.2)
    s = ReflectiveConstraintSolver(A, b)

    utils = {"m1": 0.6, "m2": 0.59}
    m_star, _ = s.select(xB, [m1, m2], utils, context={"region": "US"})
    assert m_star.name == "m1"

    # Add a positive shift to the selected model; selection should remain
    utils2 = {"m1": utils["m1"] + 0.01, "m2": utils["m2"]}
    m_star2, _ = s.select(xB, [m1, m2], utils2, context={"region": "US"})
    assert m_star2.name == "m1"

