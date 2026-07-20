from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from compitum.cli import _load_constraints, _toy_models  # type: ignore[import]
from compitum.boundary import BoundaryAnalyzer  # type: ignore[import]
from compitum.coherence import CoherenceFunctional  # type: ignore[import]
from compitum.constraints import ReflectiveConstraintSolver  # type: ignore[import]
from compitum.control import LyapunovController  # type: ignore[import]
from compitum.energy import SymbolicFreeEnergy  # type: ignore[import]
from compitum.metric import SymbolicManifoldMetric  # type: ignore[import]
from compitum.pgd import RegexPromptExtractor  # type: ignore[import]
from compitum.router import CompitumRouter  # type: ignore[import]


def build_router(D: int, defaults_path: Path, constraints_path: Path, seed: int) -> CompitumRouter:
    models = _toy_models(D)
    from compitum.predictors import CalibratedPredictor

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((128, D))
    predictors: Dict[str, Dict[str, CalibratedPredictor]] = {}
    for m in models:
        q = rng.random(128)
        t = rng.random(128)
        c = rng.random(128)
        pq = CalibratedPredictor()
        pq.fit(X, q)
        pt = CalibratedPredictor()
        pt.fit(X, t)
        pc = CalibratedPredictor()
        pc.fit(X, c)
        predictors[m.name] = {"quality": pq, "latency": pt, "cost": pc}

    import yaml

    dcfg = yaml.safe_load(Path(defaults_path).read_text())
    A, B = _load_constraints(constraints_path)
    solver = ReflectiveConstraintSolver(A, B)
    met = {
        m.name: SymbolicManifoldMetric(
            D, rank=int(dcfg["metric"]["rank"]), delta=float(dcfg["metric"]["delta"])
        )
        for m in models
    }
    coherence = CoherenceFunctional(k=64)
    boundary = BoundaryAnalyzer(
        float(dcfg.get("boundary", {}).get("gap_threshold", 0.05)),
        float(dcfg.get("boundary", {}).get("entropy_threshold", 0.65)),
        float(dcfg.get("boundary", {}).get("sigma_threshold", 0.12)),
    )
    ctrl = LyapunovController()
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
        ctrl,
        pgd,
        met,
        energy,
        update_stride=int(dcfg["update_stride"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch routing demo with tiny embeddings.")
    ap.add_argument("--D", type=int, default=35)
    ap.add_argument("--n", type=int, default=3, help="Batch size")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--defaults", type=Path, default=Path("configs/router_defaults.yaml"))
    ap.add_argument("--constraints", type=Path, default=Path("configs/constraints_us_default.yaml"))
    args = ap.parse_args()

    router = build_router(args.D, args.defaults, args.constraints, args.seed)
    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n, args.D)).astype(np.float32)
    certs = router.batch_route(X)
    out: List[Dict[str, Any]] = []
    for c in certs:
        d = json.loads(c.to_json())
        out.append(
            {
                "model": d.get("model"),
                "utility": d.get("utility"),
                "boundary_gap": d.get("boundary", {}).get("utility_gap"),
                "feasible": d.get("constraints", {}).get("feasible"),
            }
        )
    print(json.dumps({"n": len(out), "samples": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
