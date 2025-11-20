from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ...control import LyapunovController
from ...metric import SymbolicManifoldMetric


@dataclass
class PlasmaMonitorConfig:
    state_dim: int = 8
    rank: int = 4
    delta: float = 1e-3
    q_threshold: float = 1.0
    curvature_alarm: float = 0.5
    beta_d: float = 0.5
    eta: float = 0.01
    # Optional per-dimension scale factors (units normalization)
    scales: Optional[np.ndarray] = None  # None => ones
    # Optional norm exponent for distance (Lp), default 2 (Euclidean)
    norm_p: float = 2.0


class PlasmaMonitor:
    """
    Real-time stability monitor for magnetic confinement fusion.

    Maps plasma state (Te, ne, q-profile, etc.) to a geometric curvature signal
    using Compitum's SPD metric and Lyapunov-based controller.
    """

    def __init__(self, config: PlasmaMonitorConfig | None = None,
                 *,
                 state_dim: int | None = None,
                 rank: int | None = None,
                 q_threshold: float | None = None,
                 curvature_alarm: float | None = None,
                 scales: Optional[np.ndarray] = None,
                 norm_p: float | None = None) -> None:
        # Accept either a config dataclass or individual overrides for quick use
        cfg = config or PlasmaMonitorConfig()
        if state_dim is not None:
            cfg.state_dim = state_dim
        if rank is not None:
            cfg.rank = rank
        if q_threshold is not None:
            cfg.q_threshold = q_threshold
        if curvature_alarm is not None:
            cfg.curvature_alarm = curvature_alarm
        if scales is not None:
            cfg.scales = np.asarray(scales, dtype=float)
        if norm_p is not None:
            cfg.norm_p = float(norm_p)

        self.metric = SymbolicManifoldMetric(D=cfg.state_dim, rank=cfg.rank, delta=cfg.delta)
        self.controller = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
        self.equilibrium = np.zeros(cfg.state_dim)
        self.q_threshold = cfg.q_threshold
        self.curvature_alarm = cfg.curvature_alarm
        self.beta_d = cfg.beta_d
        self.eta = cfg.eta
        self.norm_p = float(cfg.norm_p)
        self.scales = (
            cfg.scales if cfg.scales is not None else np.ones(cfg.state_dim, dtype=float)
        )
        self._initialized = False

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        s = self.scales
        if s.shape != x.shape:
            # Broadcast if needed for safety; fall back to elementwise when exact shape
            s = np.broadcast_to(s, x.shape)
        return x / s

    def ingest_profile(self, state: np.ndarray, t: float) -> Dict[str, float]:
        """
        Process a single time-step of diagnostic state.

        Parameters
        - state: ndarray of shape (state_dim,) e.g., [Te_core, ne, q_min, ...]
        - t: time in milliseconds (float)

        Returns a dictionary with geometric stability signals.
        """
        if not self._initialized:
            self.equilibrium = state.copy()
            self._initialized = True

        # Apply optional units normalization
        x = self._normalize(state)
        mu = self._normalize(self.equilibrium)

        # Compute whitened residual then Lp norm if p != 2
        # Ensure metric W is ready by calling distance once (for sigma estimate)
        d2, d_std = self.metric.distance(x, mu)
        if abs(self.norm_p - 2.0) < 1e-12:
            d = d2
        else:
            # Compute whitened residual with current W and take Lp norm
            W = self.metric.W
            if W is None:
                W = self.metric._update_cholesky()  # use metric to compute W
            wz = W @ (x - mu)
            p = max(1e-6, float(self.norm_p))
            d = float(np.linalg.norm(wz, ord=p))

        grad_norm = self.metric.update_spd(
            x=x,
            mu=mu,
            beta_d=self.beta_d,
            d=d,
            eta=self.eta,
            srmf_controller=self.controller,
        )

        _, status = self.controller.update(d_star=d, grad_norm=grad_norm)
        curvature = status.get("lyapunov_function", float("nan"))
        alarm = bool(curvature > self.curvature_alarm)

        return {
            "confinement_distance": float(d),
            "confinement_distance_std": float(d_std),
            "curvature_signal": float(curvature),
            "trust_radius": float(status.get("trust_radius", float("nan"))),
            "alarm_status": alarm,
            "timestamp_ms": float(t),
        }

    def reset_equilibrium(self, new_eq: np.ndarray) -> None:
        """Re-center stability basin after a mode change or controlled crash."""
        self.equilibrium = new_eq.copy()
        self.controller.drift_integral = 0.0

