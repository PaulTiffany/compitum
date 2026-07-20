from hypothesis import given, strategies as st, settings

from compitum.cli import _load_constraints, _toy_models  # type: ignore
from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.pgd import RegexPromptExtractor
from compitum.router import CompitumRouter


def _router() -> CompitumRouter:
    D = 32
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.standard_normal((128, D))
    predictors = {}
    for m in models:
        q = rng.random(128)
        t = rng.random(128)
        c = rng.random(128)
        pq = CalibratedPredictor()
        pq.fit(X, q)
        pt = CalibratedPredictor()
        pt.fit(X, t)
        pc = CalibratedPredictor()
        pc.fit(X, c)
        predictors[m.name] = {"quality": pq, "latency": pt, "cost": pc}

    metrics = {m.name: SymbolicManifoldMetric(D, rank=6, delta=1e-3) for m in models}
    coherence = CoherenceFunctional(k=64)
    from pathlib import Path

    A, B = _load_constraints(Path("configs/constraints_us_default.yaml"))
    solver = ReflectiveConstraintSolver(A, B)
    boundary = BoundaryAnalyzer(0.05, 0.65, 0.12)
    ctrl = LyapunovController()
    energy = SymbolicFreeEnergy(0.5, 0.1, 0.1, 0.0, 0.05)
    pgd = RegexPromptExtractor()
    return CompitumRouter(
        models,
        predictors,
        solver,
        coherence,
        boundary,
        ctrl,
        pgd,
        metrics,
        energy,
        update_stride=999,
        enable_metric_update=False,
        enable_controller=False,
    )


@given(text=st.text(min_size=0, max_size=160))
@settings(deadline=None)
def test_repeated_route_is_deterministic(text: str) -> None:
    r = _router()
    # Use an explicit embedding to avoid PGD feature-length drift
    import numpy as np

    D = 32
    emb = np.zeros(D, dtype=np.float32)
    c1 = r.route(text, embedding=emb)
    c2 = r.route(text, embedding=emb)
    assert c1.model == c2.model
