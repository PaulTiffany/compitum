import numpy as np

from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model
from compitum.capabilities import Capabilities


def test_duals_monotone_near_binding_small_steps():
    A = np.array([[1.0]])
    xB = np.array([0.49])
    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    m1 = Model("m1", center=np.zeros(1), capabilities=caps, cost=0.1)
    m2 = Model("m2", center=np.zeros(1), capabilities=caps, cost=0.2)
    models = [m1, m2]
    utils = {"m1": 0.60, "m2": 0.61}

    lambdas = []
    for b in [0.49, 0.4901, 0.491, 0.495, 0.5]:
        solver = ReflectiveConstraintSolver(A, np.array([b]))
        _, info = solver.select(xB, models, utils, context={"region": "US"})
        lambdas.append(info["shadow_prices"].get("lambda_0", 0.0))
    # Non-decreasing
    assert all(lambdas[i] <= lambdas[i + 1] + 1e-12 for i in range(len(lambdas) - 1))

