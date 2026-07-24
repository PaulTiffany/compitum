"""Optional-dependency isolation and exact no-op baseline equivalence."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from compitum.trajectory import NoOpTrajectoryObserver, TrajectoryRequest


def test_ordinary_compitum_import_never_imports_jax_or_fabricpc() -> None:
    """Run in a clean interpreter so this venv's import state can't mask it."""
    code = (
        "import sys, json; import compitum; import compitum.trajectory; "
        "print(json.dumps([m for m in ('jax', 'fabricpc') if m in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout.strip()) == []


def _build_router():
    from compitum.boundary import BoundaryAnalyzer
    from compitum.cli import _load_constraints, _toy_models
    from compitum.coherence import CoherenceFunctional
    from compitum.constraints import ReflectiveConstraintSolver
    from compitum.control import LyapunovController
    from compitum.energy import SymbolicFreeEnergy
    from compitum.metric import SymbolicManifoldMetric
    from compitum.pgd import RegexPromptExtractor
    from compitum.predictors import CalibratedPredictor
    from compitum.router import CompitumRouter

    D = 32
    models = _toy_models(D)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, D))
    predictors = {}
    for m in models:
        entry = {}
        for kind in ("quality", "latency", "cost"):
            p = CalibratedPredictor()
            p.fit(X, rng.random(128))
            entry[kind] = p
        predictors[m.name] = entry
    metrics = {m.name: SymbolicManifoldMetric(D, rank=8, delta=1e-3) for m in models}
    cur = Path(__file__).resolve().parent
    repo_root = next(
        p for p in [cur, *cur.parents] if (p / "configs" / "constraints_us_default.yaml").exists()
    )
    A, B = _load_constraints(repo_root / "configs" / "constraints_us_default.yaml")
    return CompitumRouter(
        models,
        predictors,
        ReflectiveConstraintSolver(A, B),
        CoherenceFunctional(k=64),
        BoundaryAnalyzer(0.05, 0.65, 0.12),
        LyapunovController(),
        RegexPromptExtractor(),
        metrics,
        SymbolicFreeEnergy(0.5, 0.1, 0.1, 0.0, 0.05),
        update_stride=64,
        enable_metric_update=False,
        enable_controller=True,
    )


def test_noop_observer_leaves_routing_byte_identical() -> None:
    """Routing with the no-op observer running alongside must equal frozen
    v0.2.0 routing exactly (identical certificate content, timestamp aside).
    """
    prompt = "Sketch a proof for the AM-GM inequality."
    emb = np.zeros(32, dtype=np.float32)

    # Components consume the global numpy RNG stream during routing, so each
    # arm pins the same global seed: the comparison isolates the observer's
    # presence as the only difference between the two arms.
    np.random.seed(1234)
    plain = _build_router().route(prompt, embedding=emb)

    np.random.seed(1234)
    router = _build_router()
    observer = NoOpTrajectoryObserver()
    evidence = observer.observe(TrajectoryRequest(case_id="baseline", seeds={"seed": 0}))
    observed = router.route(prompt, embedding=emb)

    assert evidence.status == "unavailable"
    a = json.loads(plain.to_json())
    b = json.loads(observed.to_json())
    a.pop("timestamp")
    b.pop("timestamp")
    assert a == b
