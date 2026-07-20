import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from compitum.cli import _load_constraints, _toy_models  # type: ignore
from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.pgd import RegexPromptExtractor
from compitum.router import CompitumRouter


def _build_router(D: int = 32, enable_controller: bool = True) -> CompitumRouter:
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, D))
    predictors: Dict[str, Dict[str, CalibratedPredictor]] = {}
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
        update_stride=64,
        enable_metric_update=False,
        enable_controller=enable_controller,
    )


@pytest.mark.heavy_bench
@pytest.mark.benchmark
def test_energy_drift(benchmark) -> None:
    router = _build_router(32, enable_controller=True)
    prompts = [
        "simple query 1",
        "general query 1",
        "complex query 1",
        "simple query 2",
        "general query 2",
        "complex query 2",
    ]

    def run() -> Dict[str, int]:
        violations = 0
        energies: List[float] = []
        for p in prompts:
            emb = np.zeros(32, dtype=np.float32)
            cert = router.route(p, embedding=emb)
            energies.append(float(cert.drift_status.get("drift_ema", 0.0)))
            if len(energies) >= 2 and energies[-1] > energies[-2] + 1e-8:
                violations += 1
        return {"nonincrease_violations": violations}

    results = benchmark(run)
    # Allow a small number due to ties/float noise
    assert results["nonincrease_violations"] <= 2


@pytest.mark.heavy_bench
@pytest.mark.benchmark
def test_constraint_violation_rate(benchmark) -> None:
    router = _build_router(32, enable_controller=False)
    prompts = [
        "simple query 1",
        "general query 1",
        "complex query 1",
        "simple query 2",
        "general query 2",
        "complex query 2",
        "simple query 3",
        "general query 3",
        "complex query 3",
    ]

    def run() -> Dict[str, float]:
        violations = 0
        for p in prompts:
            emb = np.zeros(32, dtype=np.float32)
            cert = router.route(p, embedding=emb)
            feasible = bool(json.loads(cert.to_json())["constraints"]["feasible"])
            if not feasible:
                violations += 1
        return {"violation_rate": violations / max(1, len(prompts))}

    results = benchmark(run)
    # With default constraints and synthetic features, target ~0
    assert 0.0 <= results["violation_rate"] <= 0.05
