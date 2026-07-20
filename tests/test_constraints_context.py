import numpy as np

from compitum.capabilities import Capabilities
from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model


def test_select_with_context_hits_capability_branches():
    # Simple constraints: x <= b elementwise
    A = np.eye(3)
    b = np.ones(3)
    solver = ReflectiveConstraintSolver(A, b)

    # Create two models with region-gated capabilities
    caps = Capabilities(regions={"US", "EU"}, tools_allowed={"none"})
    m1 = Model(name="m1", center=np.zeros(3), capabilities=caps, cost=0.1)
    m2 = Model(name="m2", center=np.zeros(3), capabilities=caps, cost=0.2)
    models = [m1, m2]

    # Utilities rank m2 above m1 to ensure competitor loop executes
    utilities = {"m1": 0.5, "m2": 0.6}

    # Feasible x and a context specifying region so supports(..., context=...) path is taken
    xB = np.array([0.1, 0.2, 0.3])
    context = {"region": "EU"}

    chosen, info = solver.select(xB, models, utilities, context=context)

    # We should pick the higher-utility feasible model
    assert chosen.name == "m2"
    # Info dict is well-formed and indicates feasibility
    assert info["feasible"] is True
    assert info["status"] == "optimal"
    # Shadow prices present for each constraint index
    assert all(k.startswith("lambda_") for k in info["shadow_prices"].keys())
