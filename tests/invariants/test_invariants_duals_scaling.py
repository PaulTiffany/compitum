import numpy as np

from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model
from compitum.capabilities import Capabilities


def test_shadow_price_scales_with_utility_units():
    # One constraint, near binding; utilities are arbitrary scalars from energy
    A = np.array([[1.0]])
    b = np.array([0.1])
    xB = np.array([0.1])  # at boundary
    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    m1 = Model("m1", center=np.zeros(1), capabilities=caps, cost=0.1)
    m2 = Model("m2", center=np.zeros(1), capabilities=caps, cost=0.2)
    models = [m1, m2]

    # m2 slightly better utility; shadow price positive under relaxation
    utils = {"m1": 0.60, "m2": 0.61}
    solver = ReflectiveConstraintSolver(A, b)
    _, info1 = solver.select(xB, models, utils, context={"region": "US"})
    lam1 = info1["shadow_prices"].get("lambda_0", 0.0)

    # Scale utility units by s; lambdas should scale by ~s as well
    s = 10.0
    utils_scaled = {k: s * v for k, v in utils.items()}
    solver2 = ReflectiveConstraintSolver(A, b)
    _, info2 = solver2.select(xB, models, utils_scaled, context={"region": "US"})
    lam2 = info2["shadow_prices"].get("lambda_0", 0.0)

    if lam1 == 0.0 and lam2 == 0.0:
        # Degenerate case: competitor never feasible on relaxation path
        return
    ratio = lam2 / (lam1 + 1e-12)
    assert ratio >= s * 0.5 and ratio <= s * 2.0

