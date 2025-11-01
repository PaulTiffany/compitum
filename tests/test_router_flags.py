from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.capabilities import Capabilities
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.pgd import RegexPromptExtractor
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter


def _toy_models(D: int) -> list[Model]:
    rng = np.random.default_rng(7)
    centers = {
        "fast": rng.normal(0.0, 0.4, size=D),
        "thinking": rng.normal(0.0, 1.0, size=D),
        "auto": rng.normal(0.1, 0.7, size=D),
    }
    caps = Capabilities(regions={"US", "EU"}, tools_allowed={"none"})
    return [Model(name=k, center=v, capabilities=caps, cost=0.1) for k, v in centers.items()]


def _repo_root() -> Path:
    cur = Path(__file__).resolve().parent
    for p in [cur] + list(cur.parents):
        if (p / "configs" / "router_defaults.yaml").exists():
            return p
    raise RuntimeError("Could not locate repo root with configs/")


def _build_router(enable_metric_update: bool, enable_controller: bool) -> CompitumRouter:
    dcfg = yaml.safe_load((_repo_root() / "configs" / "router_defaults.yaml").read_text())
    D = int(dcfg["metric"]["D"])
    rank = int(dcfg["metric"]["rank"])
    delta = float(dcfg["metric"]["delta"])

    models = _toy_models(D)
    predictors: dict[str, dict[str, CalibratedPredictor]] = {
        m.name: {
            "quality": CalibratedPredictor(),
            "latency": CalibratedPredictor(),
            "cost": CalibratedPredictor(),
        }
        for m in models
    }

    rng = np.random.default_rng(0)
    X_demo = rng.standard_normal((128, D))
    for m in models:
        y = 0.5 + 0.1 * (X_demo @ (m.center / (1e-8 + (abs(m.center).sum() or 1.0))))
        predictors[m.name]["quality"].fit(X_demo, y)
        predictors[m.name]["latency"].fit(X_demo, abs(y))
        predictors[m.name]["cost"].fit(X_demo, 0.2 + abs(y))

    metrics = {m.name: SymbolicManifoldMetric(D, rank, delta) for m in models}
    coherence = CoherenceFunctional(k=64)
    ccfg = yaml.safe_load((_repo_root() / "configs" / "constraints_us_default.yaml").read_text())
    A, b = ccfg["A"], ccfg.get("b", ccfg.get("B"))
    solver = ReflectiveConstraintSolver(np.array(A, float), np.array(b, float))  # type: ignore[name-defined]
    boundary = BoundaryAnalyzer(0.05, 0.65, 0.12)
    controller = LyapunovController()
    energy = SymbolicFreeEnergy(dcfg["alpha"], dcfg["beta_t"], dcfg["beta_c"], dcfg["beta_d"], dcfg["beta_s"])
    pgd = RegexPromptExtractor()

    return CompitumRouter(
        models,
        predictors,
        solver,
        coherence,
        boundary,
        controller,
        pgd,
        metrics,
        energy,
        update_stride=int(dcfg["update_stride"]),
        enable_metric_update=enable_metric_update,
        enable_controller=enable_controller,
    )


def test_router_flags_disable_paths() -> None:
    router = _build_router(enable_metric_update=False, enable_controller=False)
    cert = router.route("A simple prompt.")
    assert "trust_radius" in cert.drift_status


def test_router_embedding_branch_and_batch_disable() -> None:
    dcfg = yaml.safe_load((_repo_root() / "configs" / "router_defaults.yaml").read_text())
    D = int(dcfg["metric"]["D"])
    router = _build_router(enable_metric_update=False, enable_controller=False)
    # Cover embedding path in route with correct dimension
    emb = np.zeros(D)
    cert = router.route("", embedding=emb)
    assert isinstance(cert.model, str)
    # Cover batch_route path where controller is disabled
    embs = np.zeros((3, D))
    certs = router.batch_route(embs)
    assert len(certs) == 3


def test_router_batch_metric_update_continue_branch() -> None:
    dcfg = yaml.safe_load((_repo_root() / "configs" / "router_defaults.yaml").read_text())
    D = int(dcfg["metric"]["D"])
    # Build with metric updates enabled and stride=1 to populate update_data for selected models only
    router = _build_router(enable_metric_update=True, enable_controller=True)
    router._stride = 1  # Ensure metric update phase runs
    embs = np.zeros((4, D))
    certs = router.batch_route(embs)
    # Sanity
    assert len(certs) == 4


def test_router_batch_gradnorm_update_branch_both_outcomes(monkeypatch) -> None:
    dcfg = yaml.safe_load(Path("configs/router_defaults.yaml").read_text())
    D = int(dcfg["metric"]["D"])
    router = _build_router(enable_metric_update=True, enable_controller=True)
    router._stride = 1

    # Alternate selection between first two models to ensure mixed certificates
    model_names = list(router.models.keys())
    seq = [model_names[0], model_names[1], model_names[0], model_names[1]]
    idx = {"i": 0}

    original_select = router.solver.select

    def _select(xB, models, utilities, context=None):  # type: ignore[no-redef]
        name = seq[idx["i"] % len(seq)]
        idx["i"] += 1
        # Find the model object by name
        for m in models:
            if m.name == name:
                # Return selected model and a dummy constraints info dict
                return m, {"feasible": True}
        return original_select(xB, models, utilities, context=context)

    monkeypatch.setattr(router.solver, "select", _select)

    embs = np.zeros((4, D))
    certs = router.batch_route(embs)
    assert len(certs) == 4
