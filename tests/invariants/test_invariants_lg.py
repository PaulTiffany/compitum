from __future__ import annotations

import numpy as np
import pytest

try:
    from hypothesis import assume, given, example
    from hypothesis import strategies as st
except Exception:
    pytest.skip("hypothesis not installed", allow_module_level=True)

from compitum.control import LyapunovController
from compitum.metric import SymbolicManifoldMetric
from compitum.coherence import CoherenceFunctional


def _surrogate_energy(L: np.ndarray, z: np.ndarray, beta_d: float) -> float:
    # v = L^T z, energy = 0.5 * beta_d * ||v||^2
    v = L.T @ z
    return float(0.5 * beta_d * np.sum(v * v))


@pytest.mark.invariants
@given(
    D=st.integers(2, 12),
    rank=st.integers(1, 6),
    beta_d=st.floats(1e-3, 2.0),
    eta=st.floats(1e-3, 0.5),
)
@example(D=4, rank=2, beta_d=0.5, eta=0.1)
def test_metric_update_line_search_non_increase(
    D: int, rank: int, beta_d: float, eta: float
) -> None:
    rank = min(rank, D)
    met = SymbolicManifoldMetric(D=D, rank=rank)
    ctrl = LyapunovController()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(D)
    mu = rng.standard_normal(D)
    z = x - mu
    assume(np.linalg.norm(z) > 1e-8)

    e0 = _surrogate_energy(met.L, z, beta_d)
    g = met.update_spd(
        x=x, mu=mu, beta_d=beta_d, d=float(np.linalg.norm(z)), eta=eta, srmf_controller=ctrl
    )
    assert g >= 0.0
    e1 = _surrogate_energy(met.L, z, beta_d)
    # Line search ensures non-increase (ties allowed by clipping)
    assert e1 <= e0 + 1e-12


@pytest.mark.invariants
@given(
    D=st.integers(2, 12),
    rank=st.integers(1, 6),
    beta_d=st.floats(1e-3, 2.0),
    eta=st.floats(1e-3, 0.5),
)
def test_metric_update_zero_gradient_no_change(
    D: int, rank: int, beta_d: float, eta: float
) -> None:
    rank = min(rank, D)
    met = SymbolicManifoldMetric(D=D, rank=rank)
    ctrl = LyapunovController()
    mu = np.zeros(D)
    x = mu.copy()  # z == 0
    L0 = met.L.copy()
    g = met.update_spd(x=x, mu=mu, beta_d=beta_d, d=0.0, eta=eta, srmf_controller=ctrl)
    assert g == 0.0
    assert np.allclose(met.L, L0)


@pytest.mark.invariants
def test_coherence_reservoir_capacity_and_cache() -> None:
    coh = CoherenceFunctional(k=10)
    # Before enough data, fit returns None and batch returns zeros
    kde0 = coh._fit("m")
    assert kde0 is None
    xw = np.zeros(4)
    assert np.allclose(coh.batch_log_evidence("m", np.zeros((3, 4))), 0.0)
    # Add > k items, buffer size should cap at k
    for i in range(25):
        coh.update("m", xw + i * 0.01, success=1.0)
    assert len(coh.res["m"].buf) <= 10
    kde1 = coh._fit("m")
    assert kde1 is not None
    # Update invalidates cache
    coh.update("m", xw + 1.0, success=0.5)
    assert "m" not in coh.kde_cache
