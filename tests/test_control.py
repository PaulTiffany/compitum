import numpy as np

from compitum.control import SRMFController


def test_srmf_controller_update() -> None:
    controller = SRMFController(kappa=0.1, r0=1.0)

    # Initial state
    assert controller.r == 1.0
    assert controller.ema_d == 0.0

    # 1. Test eta_cap calculation
    eta_cap, info = controller.update(d_star=1.0, grad_norm=2.0)
    assert np.isclose(eta_cap, 0.1 / (2.0 + 1e-6))
    assert np.isclose(info["trust_radius"], 1.0 * 1.1)
    assert np.isclose(info["drift_ema"], 0.1 * 1.0)

    # 2. Test trust radius decrease
    # Make ema_d high enough to trigger r decrease (ema_d > 1.5 * r)
    controller.r = 1.0
    controller.ema_d = 1.6
    _, info = controller.update(d_star=1.0, grad_norm=1.0)
    assert np.isclose(info["trust_radius"], 1.0 * 0.8)

    # 3. Test trust radius increase
    # Make ema_d low enough to trigger r increase (ema_d < 0.7 * r)
    controller.r = 1.0
    controller.ema_d = 0.6
    _, info = controller.update(d_star=1.0, grad_norm=1.0)
    assert np.isclose(info["trust_radius"], 1.0 * 1.1)

    # 4. Test trust radius clipping (upper bound)
    controller.r = 4.9
    controller.ema_d = 0.1 # will increase r
    _, info = controller.update(d_star=1.0, grad_norm=1.0)
    assert np.isclose(info["trust_radius"], 5.0) # 4.9 * 1.1 = 5.39, clipped to 5.0

    # 5. Test trust radius clipping (lower bound)
    controller.r = 0.21
    controller.ema_d = 2.0 # will decrease r
    _, info = controller.update(d_star=1.0, grad_norm=1.0)
    assert np.isclose(info["trust_radius"], 0.2) # 0.21 * 0.8 = 0.168, clipped to 0.2

    # 6. Test neutral case (no change in r)
    controller.r = 1.0
    controller.ema_d = 1.0  # 0.7 <= 1.0 <= 1.5, so no change
    _, info = controller.update(d_star=0.0, grad_norm=1.0) # d_star=0 to not change ema_d from 1.0
    assert np.isclose(info["trust_radius"], 1.0)
