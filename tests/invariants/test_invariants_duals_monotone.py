import numpy as np

from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model
from compitum.capabilities import Capabilities


def test_shadow_prices_monotone_under_relaxation():
    # With shared feasibility, shadow prices should remain zero (monotone)
    A = np.array([[1.0]])
    xB = np.array([0.1])
    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    m1 = Model("m1", center=np.zeros(1), capabilities=caps, cost=0.1)
    m2 = Model("m2", center=np.zeros(1), capabilities=caps, cost=0.2)
    models = [m1, m2]
    utils = {"m1": 0.6, "m2": 0.55}

    lambdas = []
    for b in [0.05, 0.1, 0.2, 1.0]:
        solver = ReflectiveConstraintSolver(A, np.array([b]))
        _, info = solver.select(xB, models, utils, context={"region": "US"})
        # collect lambda_0
        lambdas.append(info["shadow_prices"].get("lambda_0", 0.0))

    # Non-decreasing (will be all zeros in this setup)
    assert all(lambdas[i] <= lambdas[i + 1] + 1e-12 for i in range(len(lambdas) - 1))
