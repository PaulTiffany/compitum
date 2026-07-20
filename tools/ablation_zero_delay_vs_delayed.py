from __future__ import annotations

"""
Zero-delay vs delayed-feedback ablation for cs.SY reviewers.

Compares instantaneous updates (controller + metric each step) against a
synthetic delayed variant that batches updates every K steps. Produces a CSV and
optionally a plot if matplotlib is installed.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.pgd import RegexPromptExtractor
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter


@dataclass
class RunResult:
    distances: List[float]
    trust_radius: List[float]


def _build_single_model_router(D: int = 8, stride: int = 1) -> Tuple[CompitumRouter, Model]:
    rng = np.random.default_rng(0)
    center = rng.normal(0.0, 0.5, size=D)
    from compitum.capabilities import Capabilities

    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    model = Model(name="m", center=center, capabilities=caps, cost=0.1)

    predictors: dict[str, dict[str, CalibratedPredictor]] = {
        model.name: {
            "quality": CalibratedPredictor(),
            "latency": CalibratedPredictor(),
            "cost": CalibratedPredictor(),
        }
    }
    X = rng.standard_normal((512, D))
    y = rng.random(512)
    for key in ("quality", "latency", "cost"):
        predictors[model.name][key].fit(X, y)

    metrics = {model.name: SymbolicManifoldMetric(D, rank=min(4, D))}
    coherence = CoherenceFunctional(k=64)
    A = np.eye(4)
    b = np.ones(4)
    solver = ReflectiveConstraintSolver(A, b)
    boundary = BoundaryAnalyzer()
    controller = LyapunovController()
    energy = SymbolicFreeEnergy(alpha=0.5, beta_t=0.5, beta_c=0.5, beta_d=0.5, beta_s=0.0)
    pgd = RegexPromptExtractor()

    router = CompitumRouter(
        [model],
        predictors,
        solver,
        coherence,
        boundary,
        controller,
        pgd,
        metrics,
        energy,
        update_stride=stride,
        enable_metric_update=True,
        enable_controller=True,
    )
    return router, model


def run_instantaneous(T: int = 50, D: int = 8) -> RunResult:
    router, _ = _build_single_model_router(D=D, stride=1)
    emb = np.zeros(D)
    d_list: List[float] = []
    r_list: List[float] = []
    for _ in range(T):
        cert = router.route("", embedding=emb)
        d = -float(cert.utility_components["distance"])  # stored as negative
        d_list.append(d)
        r_list.append(router.srmf.trust_radius)
    return RunResult(distances=d_list, trust_radius=r_list)


def run_delayed(T: int = 50, D: int = 8, K: int = 5) -> RunResult:
    # Build with updates disabled; apply batched updates manually every K steps
    router, model = _build_single_model_router(D=D, stride=1)
    router.enable_controller = False
    router.enable_metric_update = False
    emb = np.zeros(D)
    d_list: List[float] = []
    r_list: List[float] = []
    batch_x: List[np.ndarray] = []
    batch_mu: List[np.ndarray] = []
    batch_d: List[float] = []
    for t in range(T):
        cert = router.route("", embedding=emb)
        d = -float(cert.utility_components["distance"])  # positive distance
        d_list.append(d)
        r_list.append(router.srmf.trust_radius)
        # Accumulate batch for delayed updates
        batch_x.append(emb.copy())
        batch_mu.append(model.center.copy())
        batch_d.append(d)
        if (t + 1) % K == 0:
            # Metric delayed update
            met = router.metric_map[model.name]
            _ = met.batch_update_spd(
                np.array(batch_x),
                np.array(batch_mu),
                router.energy.beta_d,
                np.array(batch_d),
                eta=1e-2,
                srmf_controller=router.srmf,
            )
            # Controller delayed update
            _ = router.srmf.batch_update(np.array(batch_d), np.ones(len(batch_d)))
            batch_x.clear()
            batch_mu.clear()
            batch_d.clear()
    return RunResult(distances=d_list, trust_radius=r_list)


def write_csv(out: Path, name: str, rr: RunResult) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", f"d_{name}", f"r_{name}"])
        for i, (d, r) in enumerate(zip(rr.distances, rr.trust_radius)):
            w.writerow([i, d, r])


def try_plot(out_png: Path, inst: RunResult, delay: RunResult) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(inst.distances, label="distance (instant)")
        ax[0].plot(delay.distances, label="distance (delayed)", linestyle="--")
        ax[0].set_ylabel("distance")
        ax[0].legend()
        ax[1].plot(inst.trust_radius, label="r (instant)")
        ax[1].plot(delay.trust_radius, label="r (delayed)", linestyle="--")
        ax[1].set_ylabel("trust radius")
        ax[1].set_xlabel("step")
        ax[1].legend()
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120)
    except Exception:
        # matplotlib not available; ignore
        pass


def main() -> None:
    T = int(
        np.clip(
            int(Path.cwd().joinpath("T.txt").read_text()) if Path("T.txt").exists() else 50, 10, 200
        )
    )
    inst = run_instantaneous(T=T, D=8)
    delayed = run_delayed(T=T, D=8, K=5)
    reports = Path("reports")
    write_csv(reports / "ablation_zero_delay_instant.csv", "instant", inst)
    write_csv(reports / "ablation_zero_delay_delayed.csv", "delayed", delayed)
    try_plot(reports / "ablation_zero_delay.png", inst, delayed)


if __name__ == "__main__":
    main()
