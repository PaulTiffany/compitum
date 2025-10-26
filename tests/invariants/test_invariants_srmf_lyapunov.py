from __future__ import annotations

import numpy as np
import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:
    pytest.skip("hypothesis not installed", allow_module_level=True)

from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.pgd import RegexPromptExtractor
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter


def _single_model_router(D: int, stride: int) -> tuple[CompitumRouter, np.ndarray]:
    rng = np.random.default_rng(0)
    center = rng.normal(0.0, 0.5, size=D)
    from compitum.capabilities import Capabilities

    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    model = Model(name="m", center=center, capabilities=caps, cost=0.1)

    predictors: dict[str, dict[str, CalibratedPredictor]] = {
        model.name: {
            "quality": CalibratedPredictor(),
            "latency": CalibratedPredictor(),
            "cost": CalibratedPredictor(),
        }
    }
    X = rng.standard_normal((256, D))
    y = rng.random(256)
    for key in ("quality", "latency", "cost"):
        predictors[model.name][key].fit(X, y)

    metrics = {model.name: SymbolicManifoldMetric(D, rank=min(4, D))}
    coherence = CoherenceFunctional(k=64)
    A = np.eye(4)
    b = np.ones(4)
    solver = ReflectiveConstraintSolver(A, b)
    boundary = BoundaryAnalyzer()
    controller = LyapunovController()
    energy = SymbolicFreeEnergy(alpha=0.5, beta_t=0.5, beta_c=0.5, beta_d=0.5, beta_s=0.0)
    pgd = RegexPromptExtractor()

    router = CompitumRouter(
        [model],
        predictors,
        solver,
        coherence,
        boundary,
        controller,
        pgd,
        metrics,
        energy,
        update_stride=stride,
    )
    return router, center


@pytest.mark.invariants
@given(D=st.integers(4, 12), steps=st.integers(3, 8))
def test_distance_decreases_over_updates(D: int, steps: int) -> None:
    """Under repeated updates with fixed embedding and single model, the
    whitened distance term should not increase overall (final <= initial).
    This checks the practical descent signal relevant to routing.
    """
    stride = 1
    router, _ = _single_model_router(D, stride)
    emb = np.zeros(D)

    d_first = None
    d_last = None
    for i in range(steps):
        cert = router.route("", embedding=emb)
        d = -float(cert.utility_components["distance"])  # components store negative distance
        if i == 0:
            d_first = d
        d_last = d
    assert d_first is not None and d_last is not None
    assert d_last <= d_first + 1e-9
