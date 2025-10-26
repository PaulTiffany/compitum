---
title: API Quickstart
description: Programmatic example showing how to construct a Compitum router and retrieve a routing certificate.
---

# API Quickstart

Goal

- Create a Compitum router in Python, route a prompt, and inspect the routing certificate.
- This mirrors the CLI demo but runs entirely in your code.

Example (Python)

```python
from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np

from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import SRMFController
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
    costs = {"fast": 0.1, "thinking": 0.5, "auto": 0.2}
    caps = {"regions": {"US", "CA", "EU"}, "tools_allowed": {"none"}}
    # minimal Model constructor wrapper
    return [Model(name=k, center=v, capabilities=None, cost=costs[k]) for k, v in centers.items()]


def build_router(defaults: Path, constraints: Path, *, seed: int = 12345) -> CompitumRouter:
    dcfg = yaml.safe_load(defaults.read_text())
    D = int(dcfg["metric"]["D"])  # feature dimension
    rank = int(dcfg["metric"]["rank"])  # low-rank metric factor
    delta = float(dcfg["metric"]["delta"])  # diagonal stability term

    # Models and predictors (demo fit)
    models = _toy_models(D)
    predictors: dict[str, dict[str, CalibratedPredictor]] = {
        m.name: {"quality": CalibratedPredictor(), "latency": CalibratedPredictor(), "cost": CalibratedPredictor()}
        for m in models
    }
    rng = np.random.default_rng(seed)
    X_demo = rng.standard_normal((512, D))
    for m in models:
        yq = 0.6 + 0.1 * np.tanh(X_demo @ (m.center / (np.linalg.norm(m.center) + 1e-8)))
        yt = 0.5 + 0.5 * np.abs(X_demo @ np.ones(D) / np.sqrt(D))
        yc = 0.2 + 0.4 * np.abs(X_demo @ (np.arange(D) / D))
        predictors[m.name]["quality"].fit(X_demo, yq)
        predictors[m.name]["latency"].fit(X_demo, yt)
        predictors[m.name]["cost"].fit(X_demo, yc)

    # Runtime components
    metrics = {m.name: SymbolicManifoldMetric(D, rank, delta) for m in models}
    coherence = CoherenceFunctional(k=500)
    A, b = yaml.safe_load(constraints.read_text()).values()
    solver = ReflectiveConstraintSolver(np.array(A, float), np.array(b, float))
    bcfg = dcfg.get("boundary", {})
    boundary = BoundaryAnalyzer(
        gap_threshold=float(bcfg.get("gap_threshold", 0.05)),
        entropy_threshold=float(bcfg.get("entropy_threshold", 0.65)),
        sigma_threshold=float(bcfg.get("sigma_threshold", 0.12)),
    )
    srmf = SRMFController()
    energy = SymbolicFreeEnergy(
        dcfg["alpha"], dcfg["beta_t"], dcfg["beta_c"], dcfg["beta_d"], dcfg["beta_s"]
    )
    pgd = RegexPromptExtractor()

    return CompitumRouter(
        models,
        predictors,
        solver,
        coherence,
        boundary,
        srmf,
        pgd,
        metrics,
        energy,
        update_stride=int(dcfg["update_stride"]),
    )


if __name__ == "__main__":
    defaults = Path("configs/router_defaults.yaml")
    constraints = Path("configs/constraints_us_default.yaml")
    router = build_router(defaults, constraints, seed=12345)
    cert = router.route("Sketch a proof for AM-GM inequality.")
    print(cert.to_json())  # structured certificate for logging/analysis
```

Notes

- This example mirrors the CLI demo and uses a small, deterministic synthetic fit for predictors.
- For production, replace the toy model/predictor setup with your own predictors and models.
- The certificate JSON schema is summarized in Certificate Schema.

