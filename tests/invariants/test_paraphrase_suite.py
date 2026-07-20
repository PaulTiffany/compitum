import json

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
    # Match PGD Riemannian feature dimension (35)
    D = 35
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor
    import numpy as np

    rng = np.random.default_rng(7)
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
    coherence = CoherenceFunctional(k=128)
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


def _variants(prompt: str):
    edits = [
        lambda s: s,
        lambda s: s + ".",
        lambda s: s.replace("  ", " "),
        lambda s: s.replace("the ", "the  "),
        lambda s: s.capitalize(),
        lambda s: s + "\n",
        lambda s: s.replace(",", ";"),
        lambda s: s.replace(" and ", " & "),
        lambda s: s.replace("Prove", "Show"),
    ]
    for fn in edits:
        yield fn(prompt)


def test_paraphrase_flip_budget():
    router = _router()
    base = "Prove AM-GM for nonnegative reals."
    results = []
    for p in _variants(base):
        cert = router.route(p)
        results.append(json.loads(cert.to_json()))

    chosen = [r["model"] for r in results]
    majority = max(set(chosen), key=chosen.count)
    # At least 2/3 of variants should keep the majority choice
    assert sum(1 for m in chosen if m == majority) >= (2 * len(chosen)) // 3
