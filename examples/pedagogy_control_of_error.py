"""
Pedagogy demo: Control of Error via practice (coherence) and prepared environment.

Runs a simple route, simulates "practice" by updating the coherence reservoir
near the winner's whitened vector, then re-routes and prints evidence/utility deltas.
No core code changes required.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from compitum.boundary import BoundaryAnalyzer
from compitum.capabilities import Capabilities
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.pgd import RegexPromptExtractor
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter


def build_demo_router() -> CompitumRouter:
    defaults = yaml.safe_load(Path("configs/router_defaults.yaml").read_text())
    D = int(defaults["metric"]["D"])
    rank = int(defaults["metric"]["rank"])
    delta = float(defaults["metric"]["delta"])

    rng = np.random.default_rng(7)
    centers = {
        "fast": rng.normal(0.0, 0.4, size=D),
        "thinking": rng.normal(0.0, 1.0, size=D),
        "auto": rng.normal(0.1, 0.7, size=D),
    }
    costs = {"fast": 0.1, "thinking": 0.5, "auto": 0.2}
    caps = Capabilities(regions={"US", "CA", "EU"}, tools_allowed={"none"})
    models = [Model(name=k, center=v, capabilities=caps, cost=costs[k]) for k, v in centers.items()]

    # lightweight predictors for demo
    X = rng.standard_normal((256, D))
    predictors = {}
    for m in models:
        q = 0.6 + 0.1 * np.tanh(X @ (m.center / (np.linalg.norm(m.center) + 1e-8)))
        t = 0.5 + 0.5 * np.abs(X @ np.ones(D) / np.sqrt(D))
        c = 0.2 + 0.4 * np.abs(X @ (np.arange(D) / D))
        pq = CalibratedPredictor(); pq.fit(X, q)
        pt = CalibratedPredictor(); pt.fit(X, t)
        pc = CalibratedPredictor(); pc.fit(X, c)
        predictors[m.name] = {"quality": pq, "latency": pt, "cost": pc}

    metrics = {m.name: SymbolicManifoldMetric(D, rank, delta) for m in models}
    coherence = CoherenceFunctional(k=512)
    A, b = yaml.safe_load(Path("configs/constraints_us_default.yaml").read_text()).values()
    solver = ReflectiveConstraintSolver(np.array(A, float), np.array(b, float))
    boundary = BoundaryAnalyzer(gap_threshold=0.05, entropy_threshold=0.65, sigma_threshold=0.12)
    controller = LyapunovController()
    energy = SymbolicFreeEnergy(defaults["alpha"], defaults["beta_t"], defaults["beta_c"], defaults["beta_d"], defaults["beta_s"])
    pgd = RegexPromptExtractor()
    return CompitumRouter(models, predictors, solver, coherence, boundary, controller, pgd, metrics, energy, update_stride=999, enable_metric_update=False, enable_controller=False)


def explain(cert_json: str) -> None:
    data = json.loads(cert_json)
    comps = data.get("utility_components", {})
    print("Decision:", data.get("model"), f"Utility={data.get('utility')}")
    print("Components: distance=", -float(comps.get("distance", 0.0)), "evidence=", comps.get("evidence", 0.0))
    print("Constraints:", data.get("constraints", {}))
    print("Boundary:", data.get("boundary", {}))


def main() -> None:
    router = build_demo_router()
    D = next(iter(router.metric_map.values())).D
    emb = np.zeros(D, dtype=np.float32)

    print("\nBefore practice:")
    cert0 = router.route("Prove AM-GM.", embedding=emb).to_json()
    explain(cert0)
    u0 = json.loads(cert0)["utility"]
    ev0 = json.loads(cert0)["utility_components"]["evidence"]
    winner = json.loads(cert0)["model"]

    print("\nSimulating practice near the winner in whitened space...")
    met = router.metric_map[winner]
    W = met.W if met.W is not None else met._update_cholesky()
    xw = W @ (emb - router.models[winner].center)
    rng = np.random.default_rng(0)
    for _ in range(200):
        noise = rng.normal(0.0, 0.05, size=xw.shape)
        router.coherence.update(winner, xw + noise, success=1.0)

    print("\nAfter practice:")
    cert1 = router.route("Prove AM-GM.", embedding=emb).to_json()
    explain(cert1)
    u1 = json.loads(cert1)["utility"]
    ev1 = json.loads(cert1)["utility_components"]["evidence"]
    print(f"\nDeltas: Δevidence={ev1 - ev0:+.4f}, Δutility={u1 - u0:+.4f}")

    print("\nPrepared environment (region=US -> JP -> US):")
    cUS = router.route("any", context={"region": "US"}, embedding=emb)
    cJP = router.route("any", context={"region": "JP"}, embedding=emb)
    print("US feasible:", json.loads(cUS.to_json())["constraints"]["feasible"], ", JP feasible:", json.loads(cJP.to_json())["constraints"]["feasible"])


if __name__ == "__main__":
    main()

