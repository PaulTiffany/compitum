import math
from typing import List

import numpy as np
from hypothesis import given, strategies as st, settings

from compitum.control import LyapunovController


@given(n=st.integers(min_value=1, max_value=25))
@settings(deadline=None)
def test_controller_zero_drift_non_decreasing_trust(n: int):
    c = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
    trust = []
    for _ in range(n):
        _, status = c.update(d_star=0.0, grad_norm=1.0)
        trust.append(status["trust_radius"])
    # Non-decreasing with zeros (up to clipping at upper bound)
    assert all(b >= a for a, b in zip(trust, trust[1:]))
    assert 0.2 <= trust[-1] <= 5.0


@given(
    seq=st.lists(
        st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=25,
    )
)
@settings(deadline=None)
def test_controller_eta_cap_scales_inverse_grad(seq: List[float]):
    c = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
    # check eta_cap changes inversely with grad_norm (keep d_star moderate)
    eta_caps = []
    for g in [max(0.01, v) for v in seq]:
        eta_cap, _ = c.update(d_star=0.5, grad_norm=g)
        eta_caps.append(eta_cap)
    # As grad grows, eta_cap should not increase
    for (g1, e1), (g2, e2) in zip(zip(seq, eta_caps), zip(seq[1:], eta_caps[1:])):
        if g2 > g1 + 1e-12:
            assert e2 <= e1 + 1e-9
