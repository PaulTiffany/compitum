import json
from pathlib import Path

from compitum.cli import _load_constraints, _toy_models  # type: ignore
from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.pgd import RegexPromptExtractor
from compitum.router import CompitumRouter


def _build_router() -> CompitumRouter:
    D = 64
    models = _toy_models(D)
    predictors = {
        m.name: {"quality": None, "latency": None, "cost": None} for m in models
    }
    # Cheap predictors: identity calibrators
    from compitum.predictors import CalibratedPredictor

    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, D))
    for m in models:
        q = rng.random(256)
        t = rng.random(256)
        c = rng.random(256)
        pq = CalibratedPredictor()
        pq.fit(X, q)
        pt = CalibratedPredictor()
        pt.fit(X, t)
        pc = CalibratedPredictor()
        pc.fit(X, c)
        predictors[m.name] = {"quality": pq, "latency": pt, "cost": pc}

    metrics = {m.name: SymbolicManifoldMetric(D, rank=8, delta=1e-3) for m in models}
    coherence = CoherenceFunctional(k=128)
    # Resolve constraints relative to a parent containing configs/
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
        update_stride=128,
        enable_metric_update=False,
        enable_controller=False,
    )


def test_certificate_minimal_schema():
    router = _build_router()
    import numpy as np
    emb = np.zeros(64, dtype=np.float32)
    cert = router.route("Prove the AM-GM inequality.", embedding=emb)
    data = json.loads(cert.to_json())

    # Minimal structural checks (no jsonschema dependency)
    assert set(["model", "utility", "utility_components", "constraints", "boundary", "drift"]).issubset(
        data.keys()
    )
    assert isinstance(data["model"], str)
    assert isinstance(data["utility"], (int, float))
    assert isinstance(data["utility_components"], dict)
    assert isinstance(data["constraints"], dict) and "feasible" in data["constraints"]
    assert isinstance(data["boundary"], dict)
    assert isinstance(data["drift"], dict)
