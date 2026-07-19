import numpy as np

from compitum.energy import SymbolicFreeEnergy
from compitum.coherence import CoherenceFunctional
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.capabilities import Capabilities
from compitum.predictors import CalibratedPredictor


def _constant_predictors(D: int) -> dict[str, CalibratedPredictor]:
    # Fit on a constant target so quality/latency/cost predict the same value
    # regardless of x. A target uncorrelated with X (e.g. plain noise) would let
    # the fitted model extrapolate arbitrarily at x_far, confounding the distance
    # effect this test is actually meant to isolate.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, D))
    predictors: dict[str, CalibratedPredictor] = {}
    for name in ("quality", "latency", "cost"):
        p = CalibratedPredictor()
        p.fit(X, np.full(128, 0.5))
        predictors[name] = p
    return predictors


def test_energy_monotonic_wrt_distance_and_evidence():
    D = 6
    model = Model(name="m", center=np.zeros(D), capabilities=Capabilities(regions={"US"}, tools_allowed={"none"}), cost=0.1)
    predictors = _constant_predictors(D)
    metric = SymbolicManifoldMetric(D, rank=3, delta=1e-3)
    coherence = CoherenceFunctional(k=50)

    # Seed coherence with points scattered around center so evidence is positive near, low far.
    # Reseeding default_rng(1) inside the loop would draw the same single point 20 times,
    # collapsing the "cluster" into one duplicated point instead of a real distribution.
    rng = np.random.default_rng(1)
    for _ in range(20):
        xw = rng.normal(0.0, 0.2, size=D)
        coherence.update(model.name, xw, success=1.0)

    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.5, beta_c=0.2, beta_d=0.3, beta_s=0.4)

    x_near = np.zeros(D)
    x_far = np.ones(D) * 3.0

    U_near, _, comps_near = energy.compute(x_near, model, predictors, coherence, metric)
    U_far, _, comps_far = energy.compute(x_far, model, predictors, coherence, metric)

    # Increasing distance should penalize utility
    assert U_near > U_far
    # Evidence term should be higher near cluster
    assert comps_near["evidence"] >= comps_far["evidence"]

