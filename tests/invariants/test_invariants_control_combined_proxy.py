import numpy as np

from compitum.cli import _load_constraints, _toy_models  # type: ignore
from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.pgd import RegexPromptExtractor
from compitum.router import CompitumRouter


def _router(D: int = 32) -> CompitumRouter:
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, D))
    predictors = {}
    for m in models:
        q = rng.random(128)
        t = rng.random(128)
        c = rng.random(128)
        pq = CalibratedPredictor(); pq.fit(X, q)
        pt = CalibratedPredictor(); pt.fit(X, t)
        pc = CalibratedPredictor(); pc.fit(X, c)
        predictors[m.name] = {"quality": pq, "latency": pt, "cost": pc}

    metrics = {m.name: SymbolicManifoldMetric(D, rank=8, delta=1e-3) for m in models}
    coherence = CoherenceFunctional(k=128)
    from pathlib import Path
    # Resolve constraints relative to repo root for stability across runners
    repo_root = Path(__file__).resolve().parents[2]
    A, B = _load_constraints(repo_root / "configs" / "constraints_us_default.yaml")
    solver = ReflectiveConstraintSolver(A, B)
    boundary = BoundaryAnalyzer(0.05, 0.65, 0.12)
    ctrl = LyapunovController()
    energy = SymbolicFreeEnergy(0.6, 0.1, 0.2, 0.2, 0.0)
    pgd = RegexPromptExtractor()
    return CompitumRouter(
        models, predictors, solver, coherence, boundary, ctrl, pgd, metrics, energy,
        update_stride=1, enable_metric_update=True, enable_controller=True,
    )


def test_combined_proxy_bounded_over_steps():
    r = _router(32)
    emb = np.zeros(32, dtype=np.float32)
    # Define a simple proxy: Lyapunov + scaled distance to winner
    proxy = []
    for _ in range(12):
        cert = r.route("any", embedding=emb)
        d_neg = float(cert.utility_components["distance"])  # negative distance
        d = -d_neg
        v = r.controller.lyapunov_function() + 0.01 * d
        proxy.append(v)
    # Boundedness: no explosion — ensure proxy stays finite and non-negative
    assert all(np.isfinite(proxy))
    assert min(proxy) >= 0.0
