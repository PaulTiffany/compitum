from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:
    pytest.skip("hypothesis not installed", allow_module_level=True)

from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.pgd import RegexPromptExtractor
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter


@pytest.mark.invariants
@given(
    r0=st.floats(0.2, 5.0),
    integral0=st.floats(-10.0, 10.0),
    steps=st.integers(1, 10),
    grad=st.floats(1e-6, 1e3),
)
def test_lyapunov_decays_under_zero_error(
    r0: float, integral0: float, steps: int, grad: float
) -> None:
    """With d_star == 0 over multiple steps, the Lyapunov candidate (integral^2)
    should be non-increasing due to anti-windup decay.
    """
    c = LyapunovController(kappa=0.1, r0=r0, integral_gain=0.01)
    c.drift_integral = integral0
    V_prev = c.lyapunov_function()
    for _ in range(steps):
        _, _ = c.update(d_star=0.0, grad_norm=grad)
        V_now = c.lyapunov_function()
        assert V_now <= V_prev + 1e-12
        V_prev = V_now


@pytest.mark.invariants
@given(
    n=st.integers(1, 10),
    dstars=st.lists(st.floats(0.0, 5.0), min_size=1, max_size=10),
    gnorms=st.lists(st.floats(1e-6, 1e3), min_size=1, max_size=10),
)
def test_controller_batch_equivalence_to_sequential(
    n: int, dstars: List[float], gnorms: List[float]
) -> None:
    """Batch update should be equivalent to applying single-step update sequentially."""
    # Truncate to min length
    m = min(n, len(dstars), len(gnorms))
    dstars = dstars[:m]
    gnorms = gnorms[:m]

    c_seq = LyapunovController(kappa=0.1, r0=1.0)
    c_bat = LyapunovController(kappa=0.1, r0=1.0)

    eta_seq: List[float] = []
    stat_seq: List[dict] = []
    for d, g in zip(dstars, gnorms):
        e, s = c_seq.update(d_star=d, grad_norm=g)
        eta_seq.append(e)
        stat_seq.append(s)

    e_b, s_b = c_bat.batch_update(np.array(dstars), np.array(gnorms))

    assert np.allclose(np.array(eta_seq), np.array(e_b))
    # Compare a few key fields in drift status
    for s1, s2 in zip(stat_seq, s_b):
        for k in ("trust_radius", "drift_ema", "drift_integral"):
            assert np.isclose(s1[k], s2[k])


def _build_single_model_router(
    D: int = 8, stride: int = 3
) -> Tuple[CompitumRouter, SymbolicManifoldMetric, np.ndarray]:
    rng = np.random.default_rng(0)
    center = rng.normal(0.0, 0.5, size=D)
    # Provide permissive capabilities so feasibility checks pass
    from compitum.capabilities import Capabilities

    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    model = Model(name="m", center=center, capabilities=caps, cost=0.1)

    predictors: dict[str, dict[str, CalibratedPredictor]] = {
        model.name: {
            "quality": CalibratedPredictor(),
            "latency": CalibratedPredictor(),
            "cost": CalibratedPredictor(),
        }
    }
    # Minimal fitting to allow predictions
    X = rng.standard_normal((256, D))
    y = rng.random(256)
    for key in ("quality", "latency", "cost"):
        predictors[model.name][key].fit(X, y)

    metrics = {model.name: SymbolicManifoldMetric(D, rank=min(4, D))}
    coherence = CoherenceFunctional(k=64)
    # Simple box constraints in Banach space of size 4
    A = np.eye(4)
    b = np.ones(4)
    solver = ReflectiveConstraintSolver(A, b)
    boundary = BoundaryAnalyzer()
    controller = LyapunovController()
    energy = SymbolicFreeEnergy(alpha=0.5, beta_t=0.5, beta_c=0.5, beta_d=0.5, beta_s=0.0)
    pgd = RegexPromptExtractor()

    router = CompitumRouter(
        [model],
        predictors,
        solver,
        coherence,
        boundary,
        controller,
        pgd,
        metrics,
        energy,
        update_stride=stride,
    )

    return router, metrics[model.name], center


@pytest.mark.invariants
def test_two_timescale_metric_update_stride() -> None:
    """Metric updates happen on the slower timescale; before hitting stride,
    the low-rank factor L should remain unchanged for the selected model.
    """
    router, metric, center = _build_single_model_router(D=8, stride=3)
    L0 = metric.L.copy()
    # Use zero embedding so z = -center != 0; gradient nonzero
    emb = np.zeros(8)
    router.route("", embedding=emb)
    assert np.allclose(metric.L, L0)
    router.route("", embedding=emb)
    assert np.allclose(metric.L, L0)
    router.route("", embedding=emb)
    # After hitting stride, L should update
    assert not np.allclose(metric.L, L0)


@pytest.mark.invariants
def test_controller_integral_bounded_under_bounded_drift() -> None:
    """With |d_star| bounded and decay present, the integral term remains bounded.
    This is an ISS-style sanity bound for the discrete integral with decay.
    """
    c = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
    rng = np.random.default_rng(0)
    dmax = 1.0
    for _ in range(200):
        d = float(rng.uniform(-dmax, dmax))
        c.update(d_star=d, grad_norm=1.0)
    # Geometric bound ~ dmax/(1-0.95) = 20; allow cushion
    assert abs(c.drift_integral) <= 25.0 * dmax


@pytest.mark.invariants
def test_trust_radius_moves_toward_bounds_under_persistent_signals() -> None:
    """Persistent large drift signal reduces trust radius toward lower bound;
    near-zero drift increases it toward upper bound.
    """
    # Downward trend to lower bound
    c1 = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.0)
    for _ in range(60):
        c1.update(d_star=5.0, grad_norm=1.0)
    assert c1.trust_radius <= 0.25  # near lower clip 0.2

    # Upward trend to upper bound
    c2 = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.0)
    for _ in range(60):
        c2.update(d_star=0.0, grad_norm=1.0)
    assert c2.trust_radius >= 4.5  # near upper clip 5.0
