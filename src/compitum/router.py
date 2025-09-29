from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from .boundary import BoundaryAnalyzer
from .coherence import CoherenceFunctional
from .constraints import ReflectiveConstraintSolver
from .control import SRMFController
from .energy import SymbolicFreeEnergy
from .metric import SymbolicManifoldMetric
from .models import Model
from .pgd import ProductionPGDExtractor
from .predictors import CalibratedPredictor
from .utils import pgd_hash, split_features


@dataclass
class SwitchCertificate:
    model: str
    utility: float
    utility_components: Dict[str, float]
    constraints: Dict[str, Any]
    boundary_analysis: Dict[str, Any]
    drift_status: Dict[str, float]
    pgd_signature: str
    timestamp: float
    router_version: str = "0.1.0"

    def to_json(self) -> str:
        return json.dumps({
            "model": self.model,
            "utility": round(self.utility, 6),
            "utility_components": {k: float(v) for k, v in self.utility_components.items()},
            "constraints": self.constraints,
            "boundary": self.boundary_analysis,
            "drift": self.drift_status,
            "pgd_signature": self.pgd_signature[:16],
            "timestamp": self.timestamp,
            "router_version": self.router_version
        }, indent=2)

class CompitumRouter:
    def __init__(self, models: List[Model], predictors: Dict[str, Dict[str, CalibratedPredictor]],
                 solver: ReflectiveConstraintSolver, coherence: CoherenceFunctional,
                 boundary: BoundaryAnalyzer, srmf: SRMFController,
                 pgd_extractor: ProductionPGDExtractor,
                 metric_map: Dict[str, SymbolicManifoldMetric], energy: SymbolicFreeEnergy,
                 update_stride: int = 8) -> None:
        self.models = {m.name: m for m in models}
        self.predictors = predictors
        self.solver = solver
        self.coherence = coherence
        self.boundary = boundary
        self.srmf = srmf
        self.pgd = pgd_extractor
        self.metric_map = metric_map
        self.energy = energy
        self._step = 0
        self._stride = max(int(update_stride), 1)

    # BRIDGEBLOCK_START alg:router_orchestration
    def route(self, prompt: str, context: Dict[str, Any] | None = None) -> SwitchCertificate:
        context = context or {}
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

        m_star, cinfo = self.solver.select(xB, list(self.models.values()), utilities)
        binfo = self.boundary.analyze(utilities, u_sigmas)

        # Adapt metric periodically (two-timescale)
        self._step += 1
        grad_norm = 1.0
        if self._step % self._stride == 0:
            met = self.metric_map[m_star.name]
            d_best = abs(-comps[m_star.name]["distance"])
            grad_norm = met.update_spd(xR_all, self.models[m_star.name].center, self.energy.beta_d,
                                       d_best, eta=1e-2, srmf_controller=self.srmf)

        _, drift = self.srmf.update(
            d_star=abs(-comps[m_star.name]["distance"]), grad_norm=grad_norm
        )

        cert = SwitchCertificate(
            model=m_star.name,
            utility=utilities[m_star.name],
            utility_components=comps[m_star.name],
            constraints=cinfo,
            boundary_analysis=binfo,
            drift_status=drift,
            pgd_signature=pgd_hash(prompt),
            timestamp=time.time()
        )
        return cert
    # BRIDGEBLOCK_END alg:router_orchestration
