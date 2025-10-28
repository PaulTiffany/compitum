import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

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
    D = 32
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor
    import numpy as np

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
        update_stride=128,
        enable_metric_update=False,
        enable_controller=False,
    )


def test_certificate_validates_against_jsonschema():
    r = _build_router()
    import numpy as np
    emb = np.zeros(32, dtype=np.float32)
    data = json.loads(r.route("AM-GM", embedding=emb).to_json())

    schema_path = Path("docs/_extra/assets/certificate.schema.json")
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance=data, schema=schema)

