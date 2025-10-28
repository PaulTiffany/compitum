from typing import List, Tuple

import numpy as np
from hypothesis import given, strategies as st

from compitum.control import LyapunovController


@given(
    d=st.floats(min_value=0.0, max_value=2.0),
    g=st.floats(min_value=1e-6, max_value=10.0),
)
def test_eta_cap_and_bounds(d: float, g: float) -> None:
    ctrl = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
    eta_cap, status = ctrl.update(d_star=float(d), grad_norm=float(g))
    # eta_cap follows the defined formula and is positive
    assert eta_cap > 0.0
    # trust radius stays within design bounds
    assert 0.2 <= status["trust_radius"] <= 5.0
    # Lyapunov candidate is nonnegative
    assert status["lyapunov_function"] >= 0.0


@given(
    d=st.floats(min_value=0.0, max_value=2.0),
    g1=st.floats(min_value=1e-6, max_value=5.0),
    g2=st.floats(min_value=5.0, max_value=10.0),
)
def test_eta_cap_monotone_in_grad_norm(d: float, g1: float, g2: float) -> None:
    # Fresh controllers to avoid state coupling
    c1 = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.0)
    c2 = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.0)
    e1, _ = c1.update(d_star=float(d), grad_norm=float(g1))
    e2, _ = c2.update(d_star=float(d), grad_norm=float(g2))
    assert e1 >= e2  # larger grad_norm → smaller capped step


@given(
    seq=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=2.0),
            st.floats(min_value=1e-6, max_value=10.0),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_deterministic_state_evolution(seq: List[Tuple[float, float]]) -> None:
    c1 = LyapunovController(kappa=0.2, r0=0.9, integral_gain=0.01)
    c2 = LyapunovController(kappa=0.2, r0=0.9, integral_gain=0.01)
    for d, g in seq:
        c1.update(float(d), float(g))
        c2.update(float(d), float(g))
    # Same inputs → same controller state
    assert np.isclose(c1.trust_radius, c2.trust_radius, rtol=1e-9, atol=1e-9)
