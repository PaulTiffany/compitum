import json
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


def _router(D: int = 35) -> CompitumRouter:
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor

    rng = np.random.default_rng(4)
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
    # Resolve constraints relative to a parent containing configs/ (works under mutmut)
    cur = Path(__file__).resolve().parent
    repo_root = None
    for p in [cur] + list(cur.parents):
        if (p / "configs" / "constraints_us_default.yaml").exists():
            repo_root = p
            break
    assert repo_root is not None, "Could not locate repo root with configs/"
    A, B = _load_constraints(repo_root / "configs" / "constraints_us_default.yaml")
    solver = ReflectiveConstraintSolver(A, B)
    boundary = BoundaryAnalyzer(0.05, 0.65, 0.12)
    ctrl = LyapunovController()
    energy = SymbolicFreeEnergy(0.6, 0.1, 0.2, 0.2, 0.0)
    pgd = RegexPromptExtractor()
    return CompitumRouter(
        models, predictors, solver, coherence, boundary, ctrl, pgd, metrics, energy,
        update_stride=999, enable_metric_update=False, enable_controller=False,
    )


def test_batch_determinism_same_models_twice():
    r = _router(35)
    rng = np.random.default_rng(0)
    # Fixed embeddings for determinism
    embs = [rng.standard_normal(35).astype(np.float32) * 0.0 for _ in range(6)]
    prompts = ["p0", "p1", "p2", "p3", "p4", "p5"]

    def run_batch():
        return [r.route(p, embedding=e).model for p, e in zip(prompts, embs)]

    m1 = run_batch()
    m2 = run_batch()
    assert m1 == m2
