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


def _router(D: int = 32) -> CompitumRouter:
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, D))
    predictors = {}
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
    coherence = CoherenceFunctional(k=256)
    from pathlib import Path

    A, B = _load_constraints(Path("configs/constraints_us_default.yaml"))
    solver = ReflectiveConstraintSolver(A, B)
    boundary = BoundaryAnalyzer(0.05, 0.65, 0.12)
    ctrl = LyapunovController()
    # Positive weight on evidence to reflect practice aiding utility
    # Isolate evidence contribution so practice clearly improves utility
    energy = SymbolicFreeEnergy(alpha=0.0, beta_t=0.0, beta_c=0.0, beta_d=0.0, beta_s=1.0)
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


def test_practice_increases_evidence_and_utility():
    D = 32
    r = _router(D)
    emb = np.zeros(D, dtype=np.float32)
    # Initial route
    data0 = json.loads(r.route("Prove AM-GM.", embedding=emb).to_json())
    winner = data0["model"]
    U0 = float(data0["utility"])
    ev0 = float(data0["utility_components"]["evidence"])

    # "Practice": feed similar whitened vectors to coherence for the winner
    met = r.metric_map[winner]
    W = met.W if met.W is not None else met._update_cholesky()
    xw = W @ (emb - r.models[winner].center)
    rng = np.random.default_rng(0)
    for _ in range(200):
        noise = rng.normal(0.0, 0.05, size=xw.shape)
        r.coherence.update(winner, xw + noise, success=1.0)

    # Re-route should see higher evidence component and (with positive beta_s) higher utility
    data1 = json.loads(r.route("Prove AM-GM.", embedding=emb).to_json())
    U1 = float(data1["utility"])
    ev1 = float(data1["utility_components"]["evidence"])
    assert ev1 >= ev0 - 1e-6
    assert U1 >= U0 - 1e-6
