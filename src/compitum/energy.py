from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .coherence import CoherenceFunctional
from .metric import SymbolicManifoldMetric
from .models import Model
from .predictors import CalibratedPredictor


class SymbolicFreeEnergy:
    def __init__(
        self, alpha: float, beta_t: float, beta_c: float, beta_d: float, beta_s: float
    ) -> None:
        self.alpha = alpha
        self.beta_t = beta_t
        self.beta_c = beta_c
        self._beta_d = beta_d
        self.beta_s = beta_s

    @property
    def beta_d(self) -> float: return self._beta_d
    @beta_d.setter
    def beta_d(self, v: float) -> None: self._beta_d = v

    def compute(self, xR: np.ndarray, model: Model, predictors: Dict[str, CalibratedPredictor],
               coherence: CoherenceFunctional, metric: SymbolicManifoldMetric
               ) -> Tuple[float, float, Dict[str, float]]:
        d, d_std = metric.distance(xR, model.center)
        q, q_lo, q_hi = predictors["quality"].predict(np.array([xR]))
        t, t_lo, t_hi = predictors["latency"].predict(np.array([xR]))
        c, c_lo, c_hi = predictors["cost"].predict(np.array([xR]))

        # evidence in whitened space
        W = metric.W if metric.W is not None else metric._update_cholesky()
        xw = W @ (xR - model.center)
        log_e = coherence.log_evidence(model.name, xw)

        # BRIDGEBLOCK_START def:symbolic_free_energy_computation
        U = (
            self.alpha * q[0]
            - self.beta_t * t[0]
            - self.beta_c * (c[0] + model.cost)
            - self.beta_d * d
            + self.beta_s * log_e
        )
        # BRIDGEBLOCK_END def:symbolic_free_energy_computation
        U_var = ((self.alpha*(q_hi[0]-q_lo[0])/3.92)**2 + (self.beta_t*(t_hi[0]-t_lo[0])/3.92)**2 +
                 (self.beta_c*(c_hi[0]-c_lo[0])/3.92)**2 + (self.beta_d*d_std)**2)
        comps = {
            "quality": float(q[0]), "latency": float(-t[0]), "cost": float(-(c[0] + model.cost)),
            "distance": float(-d), "evidence": float(log_e),
            "uncertainty": float(np.sqrt(U_var))
        }
        return float(U), float(np.sqrt(U_var)), comps
