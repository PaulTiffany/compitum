import itertools
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
    D = 35
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor
    import numpy as np

    rng = np.random.default_rng(1)
    X = rng.standard_normal((128, D))
    predictors = {}
    for m in models:
        q = rng.random(128)
        t = rng.random(128)
        c = rng.random(128)
        predictors[m.name] = {
            "quality": (
                lambda cp=(
                    __import__("compitum.predictors").predictors.CalibratedPredictor()
                ): cp.fit(X, q) or cp
            )(),
            "latency": (
                lambda cp=(
                    __import__("compitum.predictors").predictors.CalibratedPredictor()
                ): cp.fit(X, t) or cp
            )(),
            "cost": (
                lambda cp=(
                    __import__("compitum.predictors").predictors.CalibratedPredictor()
                ): cp.fit(X, c) or cp
            )(),
        }

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


def _variants(prompt: str):
    edits = [
        lambda s: s,
        lambda s: s + ".",
        lambda s: s.replace("  ", " "),
        lambda s: s.replace("the ", "the  "),
        lambda s: s.capitalize(),
    ]
    for fn in edits:
        yield fn(prompt)


def test_small_edits_do_not_flip_route_excessively():
    router = _router()
    base = "Prove AM-GM for nonnegative reals."
    results = []
    for p in _variants(base):
        cert = router.route(p)
        results.append(json.loads(cert.to_json()))

    chosen = [r["model"] for r in results]
    # Heuristic bound: at least half the variants pick the majority model
    majority = max(set(chosen), key=chosen.count)
    assert sum(1 for m in chosen if m == majority) >= len(chosen) // 2
