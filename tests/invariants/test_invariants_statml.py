from __future__ import annotations

import numpy as np
import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:
    pytest.skip("hypothesis not installed", allow_module_level=True)

from compitum.capabilities import Capabilities
from compitum.coherence import CoherenceFunctional
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.predictors import CalibratedPredictor


class _FixedPositivePredictor(CalibratedPredictor):
    """Deterministic predictor with positive outputs and fixed uncertainty width.

    This ensures monotonicity checks on linear coefficients are meaningful.
    """

    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        # Do not call super().__init__(); we don't need regressors here
        self.fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[override]
        n = X.shape[0]
        y = np.full(n, 0.5)
        lo = np.full(n, 0.4)
        hi = np.full(n, 0.6)
        return y, lo, hi


def _setup_common(
    D: int = 8,
) -> tuple[
    np.ndarray, Model, dict[str, CalibratedPredictor], CoherenceFunctional, SymbolicManifoldMetric
]:
    xR = np.zeros(D)
    center = np.ones(D) * 0.1
    caps = Capabilities(regions={"US", "EU"}, tools_allowed={"none"})
    model = Model(name="m", center=center, capabilities=caps, cost=1.0)
    preds: dict[str, CalibratedPredictor] = {
        "quality": _FixedPositivePredictor(),
        "latency": _FixedPositivePredictor(),
        "cost": _FixedPositivePredictor(),
    }
    coh = CoherenceFunctional(k=32)
    met = SymbolicManifoldMetric(D=D, rank=min(4, D))
    met._update_cholesky()
    return xR, model, preds, coh, met


@pytest.mark.invariants
@given(a1=st.floats(0.1, 1.0), a2=st.floats(0.1, 1.0))
def test_energy_monotonicity_alpha(a1: float, a2: float) -> None:
    """If alpha increases (holding others fixed), utility should not decrease.

    With positive quality predictions, U(alpha) is increasing in alpha.
    """
    xR, model, preds, coh, met = _setup_common()
    e1 = SymbolicFreeEnergy(alpha=a1, beta_t=0.5, beta_c=0.5, beta_d=0.5, beta_s=0.0)
    e2 = SymbolicFreeEnergy(alpha=a2, beta_t=0.5, beta_c=0.5, beta_d=0.5, beta_s=0.0)
    U1, _, _ = e1.compute(xR, model, preds, coh, met)
    U2, _, _ = e2.compute(xR, model, preds, coh, met)
    if a2 >= a1:
        assert U2 >= U1 - 1e-9
    else:
        assert U2 <= U1 + 1e-9


@pytest.mark.invariants
@given(b1=st.floats(0.1, 1.0), b2=st.floats(0.1, 1.0))
def test_energy_monotonicity_latency_penalty(b1: float, b2: float) -> None:
    """If beta_t increases (holding others fixed), utility should not increase.

    With positive latency predictions, U(beta_t) is decreasing in beta_t.
    """
    xR, model, preds, coh, met = _setup_common()
    e1 = SymbolicFreeEnergy(alpha=0.5, beta_t=b1, beta_c=0.5, beta_d=0.5, beta_s=0.0)
    e2 = SymbolicFreeEnergy(alpha=0.5, beta_t=b2, beta_c=0.5, beta_d=0.5, beta_s=0.0)
    U1, _, _ = e1.compute(xR, model, preds, coh, met)
    U2, _, _ = e2.compute(xR, model, preds, coh, met)
    if b2 >= b1:
        assert U2 <= U1 + 1e-9
    else:
        assert U2 >= U1 - 1e-9


@pytest.mark.invariants
@given(b1=st.floats(0.1, 1.0), b2=st.floats(0.1, 1.0))
def test_energy_monotonicity_cost_penalty(b1: float, b2: float) -> None:
    """If beta_c increases (holding others fixed), utility should not increase.

    With positive costs, U(beta_c) is decreasing in beta_c.
    """
    xR, model, preds, coh, met = _setup_common()
    e1 = SymbolicFreeEnergy(alpha=0.5, beta_t=0.5, beta_c=b1, beta_d=0.5, beta_s=0.0)
    e2 = SymbolicFreeEnergy(alpha=0.5, beta_t=0.5, beta_c=b2, beta_d=0.5, beta_s=0.0)
    U1, _, _ = e1.compute(xR, model, preds, coh, met)
    U2, _, _ = e2.compute(xR, model, preds, coh, met)
    if b2 >= b1:
        assert U2 <= U1 + 1e-9
    else:
        assert U2 >= U1 - 1e-9


@pytest.mark.invariants
@given(b1=st.floats(0.1, 1.0), b2=st.floats(0.1, 1.0))
def test_energy_monotonicity_distance_penalty_and_uncertainty(b1: float, b2: float) -> None:
    """If beta_d increases (holding others fixed), utility should not increase,
    and uncertainty should not decrease (since U_var grows with |beta_d|).
    """
    xR, model, preds, coh, met = _setup_common()
    e1 = SymbolicFreeEnergy(alpha=0.5, beta_t=0.5, beta_c=0.5, beta_d=b1, beta_s=0.0)
    e2 = SymbolicFreeEnergy(alpha=0.5, beta_t=0.5, beta_c=0.5, beta_d=b2, beta_s=0.0)
    U1, sig1, _ = e1.compute(xR, model, preds, coh, met)
    U2, sig2, _ = e2.compute(xR, model, preds, coh, met)
    if b2 >= b1:
        assert U2 <= U1 + 1e-9
        assert sig2 >= sig1 - 1e-9
    else:
        assert U2 >= U1 - 1e-9
        assert sig2 <= sig1 + 1e-9
