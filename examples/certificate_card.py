from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

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

    dcfg = (
        json.loads(Path(defaults_path).read_text().replace("'", '"'))
        if defaults_path.suffix == ".json"
        else None
    )
    # Fallback: YAML via CLI helper
    if dcfg is None:
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
    coherence = CoherenceFunctional(k=128)
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


def render_markdown_card(data: Dict[str, Any]) -> str:
    comps = data.get("utility_components", {})
    comps_sorted = sorted(comps.items(), key=lambda kv: abs(kv[1]), reverse=True)
    lines = []
    lines.append(f"# Certificate Card\n")
    lines.append(f"- Model: `{data.get('model')}`")
    lines.append(f"- Utility: {data.get('utility'):.4f}")
    if comps_sorted:
        lines.append("- Top components:")
        for k, v in comps_sorted[:3]:
            lines.append(f"  - {k}: {v:+.4f}")
    b = data.get("boundary", {})
    if b:
        gap = b.get("utility_gap")
        ent = b.get("entropy")
        amb = b.get("is_boundary")
        lines.append(f"- Boundary: gap={gap:.4f} entropy={ent:.4f} ambiguous={amb}")
    c = data.get("constraints", {})
    if c:
        lines.append(f"- Feasible: {c.get('feasible')}")
        if "shadow_prices" in c:
            try:
                nz = sum(1 for x in c["shadow_prices"] if abs(float(x)) > 1e-9)
                lines.append(f"- Shadow prices: {nz} non-zero")
            except Exception:
                pass
    d = data.get("drift", {})
    if d:
        tr = d.get("trust_radius")
        lines.append(f"- Trust radius: {tr}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Route a prompt and print a Markdown certificate card."
    )
    ap.add_argument("--prompt", required=True, help="Prompt to route")
    ap.add_argument("--defaults", type=Path, default=Path("configs/router_defaults.yaml"))
    ap.add_argument("--constraints", type=Path, default=Path("configs/constraints_us_default.yaml"))
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    # PGD extractor emits a fixed 35D Riemannian vector; keep D=35 for consistency.
    router = build_router(35, args.defaults, args.constraints, args.seed)
    cert = router.route(args.prompt)
    data = json.loads(cert.to_json())
    print(render_markdown_card(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
