from __future__ import annotations

# Last modified: 2025-10-07 01:00:00
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from . import __version__ as COMPITUM_VERSION
from .boundary import BoundaryAnalyzer
from .coherence import CoherenceFunctional
from .constraints import InfoDict, ReflectiveConstraintSolver
from .control import LyapunovController
from .energy import SymbolicFreeEnergy
from .metric import SymbolicManifoldMetric
from .models import Model
from .pgd import RegexPromptExtractor
from .predictors import CalibratedPredictor
from .utils import pgd_hash, split_features


@dataclass
class SwitchCertificate:
    model: str
    utility: float
    utility_components: Dict[str, float]
    constraints: InfoDict
    boundary_analysis: Dict[str, Any]
    drift_status: Dict[str, float]
    pgd_signature: str
    timestamp: float
    router_version: str = COMPITUM_VERSION

    def to_json(self) -> str:
        return json.dumps(
            {
                "model": self.model,
                "utility": round(self.utility, 6),
                "utility_components": {k: float(v) for k, v in self.utility_components.items()},
                "constraints": self.constraints,
                "boundary": self.boundary_analysis,
                "drift": self.drift_status,
                "pgd_signature": self.pgd_signature[:16],
                "timestamp": self.timestamp,
                "router_version": self.router_version,
            },
            indent=2,
        )


class CompitumRouter:
    """Routes prompts/embeddings across models and emits a certificate.

    Orchestration order:
    - feature extraction (PGD)
    - per-model utility + diagnostics (energy, uncertainty)
    - constraint-feasible selection
    - boundary analysis
    - optional metric update (two-timescale) with bounded step size
    - controller update (Lyapunov-based trust region)
    """

    def __init__(
        self,
        models: List[Model],
        predictors: Dict[str, Dict[str, CalibratedPredictor]],
        solver: ReflectiveConstraintSolver,
        coherence: CoherenceFunctional,
        boundary: BoundaryAnalyzer,
        srmf: LyapunovController,
        pgd_extractor: RegexPromptExtractor,
        metric_map: Dict[str, SymbolicManifoldMetric],
        energy: SymbolicFreeEnergy,
        update_stride: int = 8,
        enable_metric_update: bool = True,
        enable_controller: bool = True,
    ) -> None:
        self.models = {m.name: m for m in models}
        self.predictors = predictors
        self.solver = solver
        self.coherence = coherence
        self.boundary = boundary
        ctrl = srmf
        # Preferred name
        self.controller = ctrl
        # Legacy alias retained for compatibility
        self.srmf = self.controller
        self.pgd = pgd_extractor
        self.metric_map = metric_map
        self.energy = energy
        self._step = 0
        self._stride = max(int(update_stride), 1)
        self.enable_metric_update = bool(enable_metric_update)
        self.enable_controller = bool(enable_controller)

    # BRIDGEBLOCK_START alg:router_orchestration
    def route(
        self,
        prompt: str,
        context: Dict[str, Any] | None = None,
        embedding: np.ndarray | None = None,
    ) -> SwitchCertificate:
        start_time = time.time()
        context = context or {}
        if embedding is not None:
            xR_all = embedding
            xB = np.zeros(4)
        else:
            feats = self.pgd.extract_features(prompt)
            xR_all, xB = split_features(feats)
        utilities: Dict[str, float] = {}
        comps: Dict[str, Dict[str, float]] = {}
        u_sigmas: Dict[str, float] = {}

        for name, model in self.models.items():
            met = self.metric_map[name]
            U, sig, uc = self.energy.compute(
                xR_all, model, self.predictors[name], self.coherence, met
            )
            utilities[name] = float(U)
            comps[name] = uc
            u_sigmas[name] = float(sig)

        m_star, cinfo = self.solver.select(
            xB, list(self.models.values()), utilities, context=context
        )
        binfo = self.boundary.analyze(utilities, u_sigmas)

        # Adapt metric periodically (two-timescale)
        self._step += 1
        grad_norm = 1.0
        if self.enable_metric_update and (self._step % self._stride == 0):
            met = self.metric_map[m_star.name]
            d_best = abs(-comps[m_star.name]["distance"])
            grad_norm = met.update_spd(
                xR_all,
                self.models[m_star.name].center,
                self.energy.beta_d,
                d_best,
                eta=1e-2,
                srmf_controller=self.controller,
            )

        if self.enable_controller:
            _, drift = self.controller.update(
                d_star=abs(-comps[m_star.name]["distance"]), grad_norm=grad_norm
            )
        else:
            drift = {
                "trust_radius": self.controller.trust_radius,
                "drift_ema": self.controller.drift_ema,
                "drift_integral": self.controller.drift_integral,
                "lyapunov_function": self.controller.lyapunov_function(),
            }

        cert = SwitchCertificate(
            model=m_star.name,
            utility=utilities[m_star.name],
            utility_components=comps[m_star.name],
            constraints=cinfo,
            boundary_analysis=binfo,
            drift_status=drift,
            pgd_signature=pgd_hash(prompt),
            timestamp=time.time(),
        )
        if (
            self._step % 100 == 0
            and os.environ.get("COMPITUM_DEBUG_ROUTER") == "1"
        ):  # pragma: no cover - debug output only
            print(f"CompitumRouter.route took {time.time() - start_time:.4f} seconds")
        return cert

    def batch_route(
        self,
        embeddings: np.ndarray,
        prompts: List[str] | None = None,
        xB_batch: np.ndarray | None = None,
    ) -> List[SwitchCertificate]:
        start_time = time.time()
        num_samples = embeddings.shape[0]
        prompts = prompts if prompts is not None else ["" for _ in range(num_samples)]
        pgd_signatures = [pgd_hash(p) for p in prompts]  # Compute all hashes at once

        if xB_batch is None:
            xB_batch = np.zeros((num_samples, 4))

        all_utilities: Dict[str, np.ndarray] = {name: np.zeros(num_samples) for name in self.models}
        all_comps: Dict[str, List[Dict[str, float]]] = {
            name: [{} for _ in range(num_samples)] for name in self.models
        }
        all_u_sigmas: Dict[str, np.ndarray] = {name: np.zeros(num_samples) for name in self.models}

        for name, model in self.models.items():
            met = self.metric_map[name]
            # Assuming energy.batch_compute exists and returns arrays
            U_batch, sig_batch, uc_batch = self.energy.batch_compute(
                embeddings, model, self.predictors[name], self.coherence, met
            )
            all_utilities[name] = U_batch
            all_comps[name] = uc_batch
            all_u_sigmas[name] = sig_batch

        certificates: List[SwitchCertificate] = []
        update_data: Dict[str, List] = {name: [] for name in self.models}
        d_star_drift_batch: List[float] = []
        grad_norm_drift_batch: List[float] = []

        for i in range(num_samples):
            xB = xB_batch[i]
            current_utilities = {name: all_utilities[name][i] for name in self.models}
            current_comps = {name: all_comps[name][i] for name in self.models}
            current_u_sigmas = {name: all_u_sigmas[name][i] for name in self.models}

            m_star, cinfo = self.solver.select(
                xB, list(self.models.values()), current_utilities, context=None
            )
            binfo = self.boundary.analyze(current_utilities, current_u_sigmas)

            self._step += 1
            d_best = abs(-current_comps[m_star.name]["distance"])

            if self.enable_metric_update and (self._step % self._stride == 0):
                update_data[m_star.name].append(
                    (
                        embeddings[i],
                        self.models[m_star.name].center,
                        d_best,
                    )
                )

            # For drift, we always update to maintain the controller's state
            d_star_drift_batch.append(d_best)
            # We will compute grad_norm after the metric update, so use a placeholder for now
            grad_norm_drift_batch.append(1.0)

            cert = SwitchCertificate(
                model=m_star.name,
                utility=current_utilities[m_star.name],
                utility_components=current_comps[m_star.name],
                constraints=cinfo,
                boundary_analysis=binfo,
                drift_status={},
                pgd_signature=pgd_signatures[i],
                timestamp=time.time(),
            )
            certificates.append(cert)

        # Perform batch metric updates
        if self.enable_metric_update and (self._step >= self._stride):
            for model_name, data in update_data.items():
                if not data:
                    continue

                x_batch, mu_batch, d_batch = zip(*data)
                met = self.metric_map[model_name]
                grad_norm = met.batch_update_spd(
                    np.array(x_batch),
                    np.array(mu_batch),
                    self.energy.beta_d,
                    np.array(d_batch),
                    eta=1e-2,
                    srmf_controller=self.controller,
                )
                # Update the grad_norm placeholder for the drift update
                for i in range(len(certificates)):
                    if certificates[i].model == model_name:
                        grad_norm_drift_batch[i] = grad_norm

        # Perform batch drift update
        if self.enable_controller:
            _, drift_statuses = self.controller.batch_update(
                np.array(d_star_drift_batch), np.array(grad_norm_drift_batch)
            )
        else:
            # Use current state without updating controller
            drift_statuses = [
                {
                    "trust_radius": self.controller.trust_radius,
                    "drift_ema": self.controller.drift_ema,
                    "drift_integral": self.controller.drift_integral,
                    "lyapunov_function": self.controller.lyapunov_function(),
                }
                for _ in range(num_samples)
            ]

        # Update drift status in certificates
        for i in range(num_samples):
            certificates[i].drift_status = drift_statuses[i]

        if self._step % 100 == 0:  # pragma: no cover - debug output only
            print(
                f"CompitumRouter.batch_route took {time.time() - start_time:.4f} "
                f"seconds for {num_samples} samples"
            )
        return certificates

    # BRIDGEBLOCK_END alg:router_orchestration
