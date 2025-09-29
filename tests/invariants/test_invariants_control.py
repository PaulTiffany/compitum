import numpy as np
import pytest
from hypothesis import given

from compitum.control import SRMFController

from .harness import boundary_floats


@pytest.mark.invariants
@given(
    kappa=boundary_floats(0.01, 1.0),
    r0=boundary_floats(0.2, 5.0),
    d_star=boundary_floats(0, 10.0),
    grad_norm=boundary_floats(1e-7, 1e3),
)
def test_srmf_stability_invariants(
    kappa: float, r0: float, d_star: float, grad_norm: float
) -> None:
    # POWER: Bounds/Normalization, Order/Monotonicity
    # This property test checks that the SRMF controller's trust radius `r`
    # remains within its prescribed bounds [0.2, 5.0] and that the learning
    # rate `eta_cap` is non-negative.
    controller = SRMFController(kappa=kappa, r0=r0)

    # Initial state
    assert 0.2 <= controller.r <= 5.0

    # Update step
    eta_cap, info = controller.update(d_star=d_star, grad_norm=grad_norm)

    # Assert invariants
    assert eta_cap >= 0.0, "Learning rate cap must be non-negative"
    assert 0.2 <= info["trust_radius"] <= 5.0, "Trust radius escaped its bounds"
    assert np.isclose(
        info["trust_radius"], controller.r
    ), "Returned radius must match internal state"
