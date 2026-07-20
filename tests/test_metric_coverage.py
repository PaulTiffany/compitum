from __future__ import annotations

import numpy as np

from compitum.metric import SymbolicManifoldMetric
from compitum.control import LyapunovController


def test_metric_debug_distance_env_print(monkeypatch) -> None:
    monkeypatch.setenv("COMPITUM_DEBUG_METRIC", "1")
    met = SymbolicManifoldMetric(D=4, rank=2, delta=1e-3)
    x = np.ones(4)
    mu = np.zeros(4)
    d, sigma = met.distance(x, mu)
    assert d >= 0 and sigma >= 0


def test_metric_residual_queue_pruning() -> None:
    met = SymbolicManifoldMetric(D=4, rank=2, delta=1e-3)
    ctrl = LyapunovController()
    # Create a batch large enough to exceed residual cap of 100
    x_batch = np.random.randn(120, 4)
    mu_batch = np.zeros((120, 4))
    d_batch = np.abs(np.random.randn(120)) + 1e-3
    met.batch_update_spd(
        x_batch, mu_batch, beta_d=0.1, d_batch=d_batch, eta=1e-2, srmf_controller=ctrl
    )
    # Residuals queue should be pruned to at most 100
    assert len(met.whitened_residuals) <= 100


def test_metric_empty_batch_short_circuit() -> None:
    met = SymbolicManifoldMetric(D=4, rank=2, delta=1e-3)
    ctrl = LyapunovController()
    x_batch = np.zeros((0, 4))
    mu_batch = np.zeros((0, 4))
    d_batch = np.zeros(0)
    g = met.batch_update_spd(
        x_batch, mu_batch, beta_d=0.1, d_batch=d_batch, eta=1e-2, srmf_controller=ctrl
    )
    assert g == 1.0
