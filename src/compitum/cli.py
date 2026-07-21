from __future__ import annotations

import argparse
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from . import __version__ as COMPITUM_VERSION
from .boundary import BoundaryAnalyzer
from .capabilities import Capabilities
from .coherence import CoherenceFunctional
from .constraints import ReflectiveConstraintSolver
from .control import LyapunovController
from .energy import SymbolicFreeEnergy
from .metric import SymbolicManifoldMetric
from .models import Model
from .pgd import RegexPromptExtractor
from .predictors import CalibratedPredictor
from .router import CompitumRouter
from .security import (
    AuditRecord,
    file_sha256,
    git_commit_short,
    is_offline,
    redact_text,
    write_audit_record,
)
from .utils import pgd_hash


def _load_constraints(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    cfg = yaml.safe_load(path.read_text())
    return np.array(cfg["A"], float), np.array(cfg["b"], float)


def _resolve_config_path(path: Path, package_filename: str) -> Path:
    """Fall back to this package's bundled default config when the given
    (typically CWD-relative) path doesn't exist -- a `pip install`ed wheel has
    no `configs/` directory alongside it, only whatever ships inside the
    package itself."""
    if path.exists():
        return path
    return Path(str(files("compitum.configs").joinpath(package_filename)))


def _toy_models(D: int) -> List[Model]:
    rng = np.random.default_rng(7)
    centers = {
        "fast": rng.normal(0.0, 0.4, size=D),
        "thinking": rng.normal(0.0, 1.0, size=D),
        "auto": rng.normal(0.1, 0.7, size=D),
    }
    costs = {"fast": 0.1, "thinking": 0.5, "auto": 0.2}
    caps = Capabilities(regions={"US", "CA", "EU"}, tools_allowed={"none"})
    return [Model(name=k, center=v, capabilities=caps, cost=costs[k]) for k, v in centers.items()]


def route_command(args: argparse.Namespace) -> None:
    # Honor offline flag early for any optional integrations that may be added later
    if getattr(args, "offline", False):
        os.environ["COMPITUM_OFFLINE"] = "1"

    args.defaults = _resolve_config_path(args.defaults, "router_defaults.yaml")
    args.constraints = _resolve_config_path(args.constraints, "constraints_us_default.yaml")

    dcfg = yaml.safe_load(args.defaults.read_text())
    D = int(dcfg["metric"]["D"])
    rank = int(dcfg["metric"]["rank"])
    delta = float(dcfg["metric"]["delta"])

    models = _toy_models(D)
    predictors = {
        m.name: {
            "quality": CalibratedPredictor(),
            "latency": CalibratedPredictor(),
            "cost": CalibratedPredictor(),
        }
        for m in models
    }
    # quick synthetic fit for demo (deterministic)
    seed = int(getattr(args, "seed", 12345))
    # Global seed to ensure deterministic metric initializations and any
    # other global RNG usage aligns with demo reproducibility.
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    X_demo = rng.standard_normal((512, D))
    for m in models:
        yq = 0.6 + 0.1 * np.tanh(X_demo @ (m.center / np.linalg.norm(m.center) + 1e-8))
        yt = 0.5 + 0.5 * np.abs(X_demo @ np.ones(D) / np.sqrt(D))
        yc = 0.2 + 0.4 * np.abs(X_demo @ (np.arange(D) / D))
        predictors[m.name]["quality"].fit(X_demo, yq)
        predictors[m.name]["latency"].fit(X_demo, yt)
        predictors[m.name]["cost"].fit(X_demo, yc)

    metrics = {m.name: SymbolicManifoldMetric(D, rank, delta) for m in models}
    coherence = CoherenceFunctional(k=500)
    A, b = _load_constraints(args.constraints)
    solver = ReflectiveConstraintSolver(A, b)
    # Optional boundary thresholds from config
    bcfg = dcfg.get("boundary", {}) if isinstance(dcfg, dict) else {}
    boundary = BoundaryAnalyzer(
        gap_threshold=float(bcfg.get("gap_threshold", 0.05)),
        entropy_threshold=float(bcfg.get("entropy_threshold", 0.65)),
        sigma_threshold=float(bcfg.get("sigma_threshold", 0.12)),
    )
    lyapunov_controller = LyapunovController()
    energy = SymbolicFreeEnergy(
        dcfg["alpha"], dcfg["beta_t"], dcfg["beta_c"], dcfg["beta_d"], dcfg["beta_s"]
    )
    pgd = RegexPromptExtractor()

    router = CompitumRouter(
        models,
        predictors,
        solver,
        coherence,
        boundary,
        lyapunov_controller,
        pgd,
        metrics,
        energy,
        update_stride=int(dcfg["update_stride"]),
        enable_metric_update=not bool(getattr(args, "no_metric_update", False)),
        enable_controller=not bool(getattr(args, "no_controller", False)),
    )
    cert = router.route(args.prompt)
    if getattr(args, "trace", False) or getattr(args, "verbose", False):
        print(cert.to_json())
    else:
        print(json.dumps({"model": cert.model, "U": cert.utility}, indent=2))

    # Optional redacted audit record for reproducibility without storing plaintext prompts
    if getattr(args, "audit", False):
        constraints_sha = file_sha256(args.constraints)
        defaults_sha = file_sha256(args.defaults)
        prompt_redacted = redact_text(args.prompt)
        record = AuditRecord(
            version=COMPITUM_VERSION,
            offline=is_offline(),
            seed=int(getattr(args, "seed", 12345)),
            prompt={
                "hash": pgd_hash(args.prompt),
                "redaction": prompt_redacted,
            },
            config={
                "constraints_path": str(args.constraints),
                "constraints_sha256": constraints_sha,
                "defaults_path": str(args.defaults),
                "defaults_sha256": defaults_sha,
            },
            certificate=json.loads(cert.to_json()),
            commit=git_commit_short(),
        )
        out_dir = Path(getattr(args, "audit_dir", Path("artifacts/runs")))
        write_audit_record(record, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="compitum CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # route command
    route_parser = subparsers.add_parser("route", help="Route a prompt to the best model.")
    route_parser.add_argument("--prompt", required=True, help="The prompt to route.")
    route_parser.add_argument(
        "--constraints",
        type=Path,
        default=Path("configs/constraints_us_default.yaml"),
        help="Path to constraints config file.",
    )
    route_parser.add_argument(
        "--defaults",
        type=Path,
        default=Path("configs/router_defaults.yaml"),
        help="Path to router defaults config file.",
    )
    route_parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    route_parser.add_argument(
        "--trace",
        action="store_true",
        help="Emit process signals (constraints, boundary, drift, components).",
    )
    route_parser.add_argument(
        "--no-controller",
        action="store_true",
        help="Disable drift controller updates (ablations only).",
    )
    route_parser.add_argument(
        "--no-metric-update",
        action="store_true",
        help="Disable SPD metric updates (ablations only).",
    )
    route_parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (COMPITUM_OFFLINE=1).",
    )
    route_parser.add_argument(
        "--audit",
        action="store_true",
        help="Write a redacted audit record to artifacts/runs (no plaintext prompts).",
    )
    route_parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory for audit records (defaults to artifacts/runs).",
    )
    route_parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Deterministic seed for synthetic demo fit.",
    )
    route_parser.set_defaults(func=route_command)

    parser.add_argument("--version", action="version", version=f"compitum {COMPITUM_VERSION}")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
