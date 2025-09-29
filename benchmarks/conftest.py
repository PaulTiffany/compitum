"""
Benchmark fixture glue for Compitum.

We try to import your project's real fixtures if they exist (e.g. from
tests/test_orchestration_performance.py). If not, we fall back to minimal
stand-ins so the benchmark suite always runs.

Exports (used by tests):
  - router, random_router, fixed_best_router  (pytest fixtures)
  - _prompts()                                -> list[str]
  - _models(router)                           -> dict[str, Model]
  - _declared_cost_lat(model)                 -> (cost: float, latency: float)
  - _e2e_ms(routing_time_s, model)            -> ms: float
  - _promoted_choice(router, prompt, cert)    -> (certificate, best_model)
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest


# --- Try to reuse your project's existing test helpers/fixtures -------------
def _try_import_test_orchestration():
    for mod_name in (
        "tests.test_orchestration_performance",
        "test_orchestration_performance",
    ):
        try:
            return importlib.import_module(mod_name)
        except Exception:
            pass
    return None

_test_mod = _try_import_test_orchestration()

# --- Fallback minimal models/certs/router (used only if real fixtures missing)
class _Cert:
    def __init__(
        self,
        model: str = "fallback",
        utility: float = 0.5,
        constraints: Dict[str, Any] | None = None,
        boundary_analysis: Dict[str, Any] | None = None,
        drift_status: Dict[str, Any] | None = None,
    ):
        self.model = model
        self.utility = float(utility)
        self.constraints = constraints or {"feasible": True}
        self.boundary_analysis = boundary_analysis or {"is_boundary": False}
        self.drift_status = drift_status or {"trust_radius": 1.0}

class _Model:
    def __init__(self, name: str, quality: float = 0.5, cost: float = 0.5, latency: float = 0.05):
        self.name = name
        self.quality_score = float(quality)
        self.cost = float(cost)
        self.latency = float(latency)

class _PGD:
    def extract_features(self, prompt: str):
        p = (prompt or "").lower()
        if "simple" in p:
            return {"f1": 0.1, "f2": 0.2}, "simple"
        if "complex" in p:
            return {"f1": 0.8, "f2": 0.9}, "complex"
        return {"f1": 0.4, "f2": 0.5}, "general"

class _FallbackRouter:
    def __init__(self):
        self.models: Dict[str, _Model] = {
            "model_low_cost_low_quality": _Model("model_low_cost_low_quality", 0.2, 0.1, 0.01),
            "model_medium_cost_medium_quality": _Model(
                "model_medium_cost_medium_quality", 0.5, 0.5, 0.05
            ),
            "model_high_cost_high_quality": _Model(
                "model_high_cost_high_quality", 0.9, 1.0, 0.10
            ),
        }
        self.pgd = _PGD()

    def _score(self, mname: str, qtype: str) -> float:
        base = self.models[mname].quality_score
        if qtype == "complex" and "high_quality" in mname:
            base += 0.3
        if qtype == "simple" and "low_quality" in mname:
            base -= 0.1
        return float(base)

    def route(self, prompt: str) -> _Cert:
        _, qtype = self.pgd.extract_features(prompt)
        utilities = {k: self._score(k, qtype) for k in self.models}
        best = max(
            utilities.items(),
            key=lambda kv: kv[1]
        )[0]
        return _Cert(model=best, utility=utilities[best])

# ---------------------------- Pytest fixtures --------------------------------
@pytest.fixture
def router():
    """A router with .route(prompt) -> certificate and .models dict."""
    if _test_mod and hasattr(_test_mod, "mock_router"):
        return _test_mod.mock_router()  # type: ignore[attr-defined]
    return _FallbackRouter()

@pytest.fixture
def random_router(router):
    if _test_mod and hasattr(_test_mod, "simple_random_router"):
        return _test_mod.simple_random_router()  # type: ignore[attr-defined]

    rng = np.random.default_rng(0)

    class _RR:
        def __init__(self, models):
            # normalize to list of model objects
            self.models = list(models.values()) if isinstance(models, dict) else list(models)
            self.pgd = _PGD()

        def route(self, prompt: str):
            m = rng.choice(self.models)
            return _Cert(model=m.name, utility=getattr(m, "quality_score", 0.5))

    return _RR(getattr(router, "models", {}))

@pytest.fixture
def fixed_best_router(router):
    if _test_mod and hasattr(_test_mod, "simple_fixed_router"):
        return _test_mod.simple_fixed_router()  # type: ignore[attr-defined]

    class _FR:
        def __init__(self, models):
            self.models = list(models.values()) if isinstance(models, dict) else list(models)
            # best by quality_score (fallback to cost if missing)
            def q(m): return getattr(m, "quality_score", getattr(m, "quality", 0.0))
            self.models.sort(key=q)
            self.best = self.models[-1]
            self.pgd = _PGD()

        def route(self, prompt: str):
            u = getattr(self.best, "quality_score", getattr(self.best, "quality", 0.5))
            return _Cert(model=self.best.name, utility=float(u))

    return _FR(getattr(router, "models", {}))

# ------------------------------ Helpers exported -----------------------------
def query_stream() -> List[str]:
    return [
        "simple query 1", "general query 1", "complex query 1",
        "simple query 2", "general query 2", "complex query 2",
        "simple query 3", "general query 3", "complex query 3",
    ]

def _prompts() -> List[str]:
    """Kept separate so tests can import this exact stream."""
    return query_stream()

def _models(router) -> Dict[str, Any]:
    if hasattr(router, "models"):
        if isinstance(router.models, dict):
            return router.models
        elif isinstance(router.models, list):
            return {m.name: m for m in router.models}
    return {}

def _declared_cost_lat(model) -> Tuple[float, float]:
    """
    Extract declared cost and latency from a model.
    Returns (cost, latency) tuple.
    """
    if model is None:
        return 0.0, 0.0
    cost = getattr(model, "cost", 0.0)
    latency = getattr(model, "latency", 0.0)
    return float(cost), float(latency)

def _e2e_ms(routing_time_s: float, model) -> float:
    """
    Calculate end-to-end time in milliseconds.
    Includes routing time plus model latency.
    """
    if model is None:
        return float(routing_time_s * 1000.0)
    model_latency = getattr(model, "latency", 0.0)
    return float((routing_time_s + model_latency) * 1000.0)

def _promoted_choice(router, prompt: str, cert) -> Tuple[Any, Any]:
    """
    Given an initial certificate `cert`, return a tuple of:
      (possibly-updated certificate that uses the best model, best_model_obj).

    "Best" is chosen by highest available `quality_score` (fall back to `quality`,
    then to lowest cost if no quality metric is available).
    """
    models_dict = _models(router)
    if not models_dict:
        # No models to promote to; just return the original
        return cert, None

    def quality_key(m):
        # Prefer higher quality_score/quality; if neither, prefer lower cost as proxy
        q = getattr(m, "quality_score", getattr(m, "quality", None))
        if q is not None:
            return (1, float(q))  # tier 1: quality present
        # tier 0: no quality -> invert cost ranking via negative cost
        return (0, -float(getattr(m, "cost", 0.0)))

    best_model = max(models_dict.values(), key=quality_key)

    # Try to reuse the incoming certificate object; otherwise create a new one
    try:
        setattr(cert, "model", getattr(best_model, "name", "best"))
        setattr(cert, "utility", float(getattr(best_model, "quality_score",
                                              getattr(best_model, "quality", 0.5))))
        return cert, best_model
    except Exception:
        # If cert is immutable / not our type, create a fresh fallback certificate
        return _Cert(
            model=getattr(best_model, "name", "best"),
            utility=float(getattr(best_model, "quality_score",
                                  getattr(best_model, "quality", 0.5))),
        ), best_model
