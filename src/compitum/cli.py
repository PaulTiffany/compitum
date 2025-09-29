from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from .boundary import BoundaryAnalyzer
from .capabilities import Capabilities
from .coherence import CoherenceFunctional
from .constraints import ReflectiveConstraintSolver
from .control import SRMFController
from .energy import SymbolicFreeEnergy
from .metric import SymbolicManifoldMetric
from .models import Model
from .pgd import ProductionPGDExtractor
from .predictors import CalibratedPredictor
from .router import CompitumRouter


def _load_constraints(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    cfg = yaml.safe_load(path.read_text())
    return np.array(cfg["A"], float), np.array(cfg["b"], float)

def _toy_models(D: int) -> List[Model]:
    rng = np.random.default_rng(7)
    centers = {
        "fast":    rng.normal(0.0, 0.4, size=D),
        "thinking":rng.normal(0.0, 1.0, size=D),
        "auto":    rng.normal(0.1, 0.7, size=D)
    }
    costs = {"fast": 0.1, "thinking": 0.5, "auto": 0.2}
    caps = Capabilities(regions={"US","CA","EU"}, tools_allowed={"none"})
    return [Model(name=k, center=v, capabilities=caps, cost=costs[k]) for k, v in centers.items()]

def route_command(args: argparse.Namespace) -> None:
    dcfg = yaml.safe_load(args.defaults.read_text())
    D = int(dcfg["metric"]["D"])
    rank = int(dcfg["metric"]["rank"])
    delta = float(dcfg["metric"]["delta"])

    models = _toy_models(D)
    predictors = {
        m.name: {
            "quality": CalibratedPredictor(),
            "latency": CalibratedPredictor(),
            "cost": CalibratedPredictor()
        }
        for m in models
    }
    # quick synthetic fit for demo
    X_demo = np.random.randn(512, D)
    for m in models:
        yq = 0.6 + 0.1*np.tanh(X_demo @ (m.center/np.linalg.norm(m.center)+1e-8))
        yt = 0.5 + 0.5*np.abs(X_demo @ np.ones(D)/np.sqrt(D))
        yc = 0.2 + 0.4*np.abs(X_demo @ (np.arange(D)/D))
        predictors[m.name]["quality"].fit(X_demo, yq)
        predictors[m.name]["latency"].fit(X_demo, yt)
        predictors[m.name]["cost"].fit(X_demo, yc)

    metrics = {m.name: SymbolicManifoldMetric(D, rank, delta) for m in models}
    coherence = CoherenceFunctional(k=500)
    A,b = _load_constraints(args.constraints)
    solver = ReflectiveConstraintSolver(A, b)
    boundary = BoundaryAnalyzer()
    srmf = SRMFController()
    energy = SymbolicFreeEnergy(
        dcfg["alpha"], dcfg["beta_t"], dcfg["beta_c"], dcfg["beta_d"], dcfg["beta_s"]
    )
    pgd = ProductionPGDExtractor()

    router = CompitumRouter(
        models, predictors, solver, coherence, boundary, srmf, pgd, metrics, energy,
        update_stride=int(dcfg["update_stride"])
    )
    cert = router.route(args.prompt)
    print(
        cert.to_json() if args.verbose else json.dumps(
            {"model": cert.model, "U": cert.utility}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="compitum CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # route command
    route_parser = subparsers.add_parser("route", help="Route a prompt to the best model.")
    route_parser.add_argument("--prompt", required=True, help="The prompt to route.")
    route_parser.add_argument(
        "--constraints", type=Path, default=Path("configs/constraints_us_default.yaml"),
        help="Path to constraints config file."
    )
    route_parser.add_argument(
        "--defaults", type=Path, default=Path("configs/router_defaults.yaml"),
        help="Path to router defaults config file."
    )
    route_parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    route_parser.set_defaults(func=route_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
