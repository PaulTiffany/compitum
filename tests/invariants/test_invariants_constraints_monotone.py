import numpy as np

from compitum.constraints import ReflectiveConstraintSolver
from compitum.models import Model
from compitum.capabilities import Capabilities


def test_feasibility_monotone_in_b():
    # A x <= b; relaxing b increases feasibility
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    xB = np.array([1.0, 1.0])
    b_tight = np.array([0.5, 0.5])
    b_relaxed = np.array([2.0, 2.0])

    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    m = Model("m", center=np.zeros(1), capabilities=caps, cost=0.1)

    s_tight = ReflectiveConstraintSolver(A, b_tight)
    s_relax = ReflectiveConstraintSolver(A, b_relaxed)

    # Minimal utilities for select
    utils = {"m": 1.0}

    _, info_tight = s_tight.select(xB, [m], utils, context={"region": "US"})
    _, info_relax = s_relax.select(xB, [m], utils, context={"region": "US"})
    # If tight is feasible, relaxed must be feasible; if tight infeasible, relaxed could become feasible
    assert (not info_tight["feasible"]) or info_relax["feasible"]

