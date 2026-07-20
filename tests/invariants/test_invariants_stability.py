from __future__ import annotations

import numpy as np

from compitum.metric import SymbolicManifoldMetric
from compitum.control import LyapunovController


def _surrogate_energy(L: np.ndarray, z: np.ndarray, beta_d: float) -> float:
    # J_surr(L) = (beta_d/2) * || L^T z ||^2
    v = L.T @ z
    return float(0.5 * beta_d * float(v.T @ v))


def test_metric_update_monotone_surrogate_descent() -> None:
    rng = np.random.default_rng(42)
    D, rank = 8, 3
    met = SymbolicManifoldMetric(D, rank, delta=1e-3)
    # Initialize L deterministically for test stability
    met.L = rng.normal(0.0, 0.1, size=(D, rank))
    mu = rng.normal(0.0, 0.1, size=D)
    x = rng.normal(0.0, 1.0, size=D)
    z = x - mu
    beta_d = 0.15
    # Compute surrogate distance d approximately using current metric
    met._update_cholesky()
    d, _ = met.distance(x, mu)
    srmf = LyapunovController(kappa=0.5)
    e0 = _surrogate_energy(met.L, z, beta_d)
    # Use a relatively large user step; stability cap inside should enforce safe step
    met.update_spd(x, mu, beta_d=beta_d, d=d, eta=10.0, srmf_controller=srmf)
    e1 = _surrogate_energy(met.L, z, beta_d)
    # Monotone (non-increasing) surrogate energy
    assert e1 <= e0 + 1e-12
