import time
from typing import Any, Dict, List, Tuple

import numpy as np


# --- Copied from benchmarks/conftest.py for mocking ---
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
# --- End copied from benchmarks/conftest.py ---

class _FallbackRouter:
    def __init__(self):
        self.models: Dict[str, _Model] = {
            "model_cheap_fast_low_quality": _Model("model_cheap_fast_low_quality", 0.2, 0.1, 0.01),
            "model_medium_medium_medium_quality": _Model(
                "model_medium_medium_medium_quality", 0.5, 0.5, 0.05
            ),
            "model_expensive_slow_high_quality": _Model(
                "model_expensive_slow_high_quality", 0.9, 1.0, 0.10
            ),
            "model_mid_cost_high_quality_slow": _Model(
                "model_mid_cost_high_quality_slow", 0.8, 0.6, 0.15
            ),
            "model_high_cost_mid_quality_fast": _Model(
                "model_high_cost_mid_quality_fast", 0.6, 0.8, 0.02
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

        # Simulate Compitum's intelligent routing:
        # Find the best model overall
        overall_best_model_name = max(utilities.items(), key=lambda kv: kv[1])[0]
        overall_best_utility = utilities[overall_best_model_name]

        # Try to find a cheaper model that still meets a high utility threshold
        # (e.g., 80% of overall best)
        candidate_models = []
        for name, utility in utilities.items():
            if utility >= (overall_best_utility * 0.8): # Meets 80% of best utility
                candidate_models.append(
                    (name, utility, self.models[name].cost, self.models[name].latency)
                )

        if candidate_models:
            # Pick the cheapest among the candidates that meet the utility threshold
            chosen_model_name = min(candidate_models, key=lambda x: x[2])[0] # x[2] is cost
            return _Cert(model=chosen_model_name, utility=utilities[chosen_model_name])
        else:
            # Fallback to overall best if no cheaper alternative meets threshold
            return _Cert(model=overall_best_model_name, utility=overall_best_utility)

def _route(router, prompt: str):
    """
    Run a single routing call and capture:
      - certificate returned by router
      - the Model object actually used (looked up by name)
      - routing time in seconds
    """
    t0 = time.perf_counter()
    cert = router.route(prompt)
    dt = time.perf_counter() - t0
    mdl = _models(router).get(getattr(cert, "model", None))
    return cert, mdl, dt

def get_iso_utility_data_for_summary(router, fixed_best_router):
    taus = [0.6, 0.7, 0.8]  # utility targets
    prompts = _prompts()

    out = {}
    for tau in taus:
        # Compitum policy: pick normally; if u < tau, "promote" to best
        comp_costs, comp_e2e = [], []
        fixed_costs, fixed_e2e = [], []

        for p in prompts:
            # Compitum route
            cert_c, mdl_c, dt_c = _route(router, p)
            u_c = float(getattr(cert_c, "utility", 0.5))
            model_used = mdl_c
            if u_c < tau:
                # simulate elevation to strongest expert
                cert_c, model_used = _promoted_choice(router, p, cert_c)
            c_cost, _ = _declared_cost_lat(model_used)
            comp_costs.append(c_cost)
            comp_e2e.append(_e2e_ms(dt_c, model_used))

            # Fixed-best route
            cert_f, mdl_f, dt_f = _route(fixed_best_router, p)
            f_cost, _ = _declared_cost_lat(mdl_f)
            fixed_costs.append(f_cost)
            fixed_e2e.append(_e2e_ms(dt_f, mdl_f))

        comp_cost_mean = float(np.mean(comp_costs))
        fixed_cost_mean = float(np.mean(fixed_costs))
        comp_e2e_mean = float(np.mean(comp_e2e))
        fixed_e2e_mean = float(np.mean(fixed_e2e))

        savings_cost_pct = 100.0 * (fixed_cost_mean - comp_cost_mean) / max(1e-9, fixed_cost_mean)
        savings_e2e_pct = 100.0 * (fixed_e2e_mean - comp_e2e_mean) / max(1e-9, fixed_e2e_mean)

        out[f"tau_{tau}"] = {
            "comp_cost_mean": comp_cost_mean,
            "fixed_cost_mean": fixed_cost_mean,
            "savings_cost_pct": savings_cost_pct,
            "comp_e2e_mean_ms": comp_e2e_mean,
            "fixed_e2e_mean_ms": fixed_e2e_mean,
            "savings_e2e_pct": savings_e2e_pct,
        }
    return out
