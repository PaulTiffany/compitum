from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class SRMFController:
    def __init__(self, kappa: float = 0.1, r0: float = 1.0):
        self.kappa = kappa
        self.r = r0
        self.ema_d = 0.0

    # BRIDGEBLOCK_START alg:srmf_update
    def update(self, d_star: float, grad_norm: float) -> Tuple[float, Dict[str, float]]:
        self.ema_d = 0.9*self.ema_d + 0.1*float(d_star)
        eta_cap = self.kappa / (float(grad_norm) + 1e-6)
        if self.ema_d > 1.5*self.r:
            self.r *= 0.8
        elif self.ema_d < 0.7*self.r:
            self.r *= 1.1
        self.r = float(np.clip(self.r, 0.2, 5.0))
        return float(eta_cap), {"trust_radius": self.r, "drift_ema": self.ema_d}
    # BRIDGEBLOCK_END alg:srmf_update
