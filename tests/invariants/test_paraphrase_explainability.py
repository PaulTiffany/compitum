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


def _router(D: int = 35) -> CompitumRouter:
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor
    import numpy as np

    rng = np.random.default_rng(3)
    X = rng.standard_normal((256, D))
    predictors = {}
    for m in models:
        q = rng.random(256)
        t = rng.random(256)
        c = rng.random(256)
        pq = CalibratedPredictor(); pq.fit(X, q)
        pt = CalibratedPredictor(); pt.fit(X, t)
        pc = CalibratedPredictor(); pc.fit(X, c)
        predictors[m.name] = {"quality": pq, "latency": pt, "cost": pc}

    metrics = {m.name: SymbolicManifoldMetric(D, rank=8, delta=1e-3) for m in models}
    coherence = CoherenceFunctional(k=128)
    from pathlib import Path
    A, B = _load_constraints(Path("configs/constraints_us_default.yaml"))
    solver = ReflectiveConstraintSolver(A, B)
    boundary = BoundaryAnalyzer(0.05, 0.65, 0.12)
    ctrl = LyapunovController()
    energy = SymbolicFreeEnergy(0.6, 0.1, 0.2, 0.2, 0.1)
    pgd = RegexPromptExtractor()
    return CompitumRouter(
        models, predictors, solver, coherence, boundary, ctrl, pgd, metrics, energy,
        update_stride=999, enable_metric_update=False, enable_controller=False,
    )


def _variants(prompt: str):
    edits = [
        lambda s: s,
        lambda s: s + ".",
        lambda s: s.replace("Prove", "Show"),
        lambda s: s.replace(" and ", " & "),
        lambda s: s.replace(",", ";"),
        lambda s: s.capitalize(),
    ]
    for fn in edits:
        yield fn(prompt)


def test_paraphrase_flips_are_explainable():
    r = _router()
    base = "Prove AM-GM for nonnegative reals."
    import numpy as np
    emb = np.zeros(35, dtype=np.float32)

    def run(text: str):
        return json.loads(r.route(text, embedding=emb).to_json())

    c0 = run(base)
    flips = 0
    for p in _variants(base):
        c1 = run(p)
        if c1["model"] != c0["model"]:
            flips += 1
            # Explainability: require a notable change in distance or feasibility
            d0 = float(c0["utility_components"]["distance"])  # negative distance
            d1 = float(c1["utility_components"]["distance"])  # negative distance
            feas0 = bool(c0["constraints"]["feasible"])
            feas1 = bool(c1["constraints"]["feasible"]) 
            assert (abs(d1 - d0) > 1e-3) or (feas0 != feas1)

