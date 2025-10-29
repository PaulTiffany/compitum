import numpy as np

from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.coherence import CoherenceFunctional
from compitum.models import Model
from compitum.predictors import CalibratedPredictor
from compitum.capabilities import Capabilities


def _const_predictor(val: float) -> CalibratedPredictor:
    cp = CalibratedPredictor()
    X = np.zeros((32, 8))
    y = np.full(32, val, dtype=float)
    cp.fit(X, y)
    return cp


def test_energy_decreases_with_distance_and_cost_increases():
    D = 8
    met = SymbolicManifoldMetric(D, rank=4, delta=1e-3)
    coh = CoherenceFunctional(k=1000)
    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    model = Model("m", center=np.zeros(D), capabilities=caps, cost=0.2)
    predictors = {
        "quality": _const_predictor(0.6),
        "latency": _const_predictor(0.3),
        "cost": _const_predictor(0.4),
    }
    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.5, beta_d=0.7, beta_s=0.0)

    x0 = np.zeros(D)
    U0, _, comps0 = energy.compute(x0, model, predictors, coh, met)
    x1 = np.zeros(D); x1[0] = 1.0
    U1, _, comps1 = energy.compute(x1, model, predictors, coh, met)

    # Distance increases (comps uses negative distance) → value becomes more negative
    assert comps0["distance"] >= comps1["distance"] - 1e-9
    assert U1 < U0

    # Cost increase lowers utility by beta_c * delta_cost
    energy2 = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.5, beta_d=0.0, beta_s=0.0)
    Uc0, _, _ = energy2.compute(x0, model, predictors, coh, met)
    model2 = Model("m2", center=np.zeros(D), capabilities=caps, cost=model.cost + 0.1)
    Uc1, _, _ = energy2.compute(x0, model2, predictors, coh, met)
    assert abs((Uc0 - Uc1) - 0.5 * 0.1) < 1e-6
