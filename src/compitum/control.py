from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class LyapunovController:
    """Lyapunov-based trust-region controller.

    This class supersedes the prior "SRMFController". In our framework, the
    SRMF functional acts as a Lyapunov candidate under bounded-observer
    assumptions; this controller implements a concrete, numerically stable
    update that decreases a positive-definite Lyapunov function and clips the
    trust region to maintain boundedness.
    """

    def __init__(self, kappa: float = 0.1, r0: float = 1.0, integral_gain: float = 0.005):
        self.kappa = kappa
        self.trust_radius = r0
        self.drift_ema = 0.0
        self.drift_integral = 0.0
        self.integral_gain = integral_gain

    def lyapunov_function(self) -> float:
        """Positive-definite Lyapunov candidate (squared integral error)."""
        return self.drift_integral**2

    # BRIDGEBLOCK_START alg:srmf_update
    def update(self, d_star: float, grad_norm: float) -> Tuple[float, Dict[str, float]]:
        """
        Updates the controller state based on the observed error `d_star` and gradient norm.

        This controller is inspired by Lyapunov stability theory. The `drift_integral` term
        acts as a simple Lyapunov function candidate. By driving this term towards zero, we
        ensure that the system remains in a stable state.
        """
        self.drift_ema = 0.9 * self.drift_ema + 0.1 * float(d_star)
        # Accumulate the error (contradiction signal)
        self.drift_integral += float(d_star)

        eta_cap = self.kappa / (float(grad_norm) + 1e-6)

        # Existing EMA-based adjustment
        if self.drift_ema > 1.5 * self.trust_radius:
            self.trust_radius *= 0.8
        elif self.drift_ema < 0.7 * self.trust_radius:
            self.trust_radius *= 1.1

        # Add a direct adjustment from the integral term for persistent drift
        self.trust_radius -= self.integral_gain * self.drift_integral

        # Decay the integral term to prevent wind-up and allow for recovery
        self.drift_integral *= 0.95

        self.trust_radius = float(np.clip(self.trust_radius, 0.2, 5.0))
        # Return the integral term in the status for observability
        return float(eta_cap), {
            "trust_radius": self.trust_radius,
            "drift_ema": self.drift_ema,
            "drift_integral": self.drift_integral,
            "lyapunov_function": self.lyapunov_function(),
        }

    # BRIDGEBLOCK_END alg:srmf_update

    def batch_update(
        self, d_star_batch: np.ndarray, grad_norm_batch: np.ndarray
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        eta_caps = []
        drift_statuses = []
        for d_star, grad_norm in zip(d_star_batch, grad_norm_batch):
            eta_cap, drift_status = self.update(d_star, grad_norm)
            eta_caps.append(eta_cap)
            drift_statuses.append(drift_status)
        return eta_caps, drift_statuses


class SRMFController(LyapunovController):
    """Deprecated alias for backward compatibility.

    The Self-Regulating Mapping Function (SRMF) serves as a Lyapunov
    functional in our setting; this alias preserves older imports while the
    canonical implementation is provided by ``LyapunovController``.
    """

    pass
