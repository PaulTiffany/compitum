from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ...control import LyapunovController
from ...metric import SymbolicManifoldMetric


@dataclass
class SuperconMonitorConfig:
    state_dim: int = 8
    rank: int = 4
    delta: float = 1e-3
    alarm_threshold: float = 0.5
    beta_d: float = 0.5
    eta: float = 0.01
    scales: Optional[np.ndarray] = None
    norm_p: float = 2.0


class SuperconMonitor:
    """
    Simulated superconductivity monitor built on Compitum's SPD metric and Lyapunov controller.

    Treats feature vectors as a point in material feature space; produces a pairing/curvature proxy
    and an alarm if the proxy exceeds a threshold. This is a simulated example (no real data).
    """

    def __init__(self, config: SuperconMonitorConfig | None = None,
                 *, state_dim: int | None = None, rank: int | None = None,
                 alarm_threshold: float | None = None,
                 scales: Optional[np.ndarray] = None,
                 norm_p: float | None = None) -> None:
        cfg = config or SuperconMonitorConfig()
        if state_dim is not None:
            cfg.state_dim = state_dim
        if rank is not None:
            cfg.rank = rank
        if alarm_threshold is not None:
            cfg.alarm_threshold = alarm_threshold
        if scales is not None:
            cfg.scales = np.asarray(scales, dtype=float)
        if norm_p is not None:
            cfg.norm_p = float(norm_p)

        self.metric = SymbolicManifoldMetric(D=cfg.state_dim, rank=cfg.rank, delta=cfg.delta)
        self.controller = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
        self.scales = cfg.scales if cfg.scales is not None else np.ones(cfg.state_dim, dtype=float)
        self.norm_p = float(cfg.norm_p)
        self.alarm_threshold = float(cfg.alarm_threshold)
        self.center = np.zeros(cfg.state_dim)
        self._initialized = False

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        s = self.scales
        if s.shape != x.shape:
            s = np.broadcast_to(s, x.shape)
        return x / s

    def ingest_features(self, x: np.ndarray, *, t: float = 0.0) -> Dict[str, float]:
        if not self._initialized:
            self.center = x.copy()
            self._initialized = True
        xn = self._normalize(x)
        cn = self._normalize(self.center)

        d2, d_std = self.metric.distance(xn, cn)
        if abs(self.norm_p - 2.0) < 1e-12:
            d = d2
        else:
            W = self.metric.W
            if W is None:
                W = self.metric._update_cholesky()
            wz = W @ (xn - cn)
            d = float(np.linalg.norm(wz, ord=max(1e-6, float(self.norm_p))))

        grad_norm = self.metric.update_spd(
            x=xn, mu=cn, beta_d=0.5, d=d, eta=0.01, srmf_controller=self.controller
        )
        _, status = self.controller.update(d_star=d, grad_norm=grad_norm)
        pairing_proxy = float(status.get("lyapunov_function", 0.0))
        alarm = pairing_proxy > self.alarm_threshold
        return {
            "distance": float(d),
            "distance_std": float(d_std),
            "pairing_proxy": pairing_proxy,
            "trust_radius": float(status.get("trust_radius", float("nan"))),
            "alarm_status": bool(alarm),
            "timestamp_ms": float(t),
        }

    def reset_center(self, new_center: np.ndarray) -> None:
        self.center = new_center.copy()
        self.controller.drift_integral = 0.0

