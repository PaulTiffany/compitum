import numpy as np

from compitum.control import LyapunovController


def test_srmf_controller_update() -> None:
    controller = LyapunovController(kappa=0.1, r0=1.0)

    # Initial state
    assert controller.trust_radius == 1.0
    assert controller.drift_ema == 0.0

    # 1. Test eta_cap calculation
    controller = LyapunovController(kappa=0.1, r0=1.0)  # Reset controller for each case
    eta_cap, info = controller.update(d_star=1.0, grad_norm=2.0)
    assert np.isclose(eta_cap, 0.1 / (2.0 + 1e-6))
    assert np.isclose(info["trust_radius"], 1.0 * 1.1 - controller.integral_gain * 1.0)
    assert np.isclose(info["drift_ema"], 0.1 * 1.0)

    # 2. Test trust radius decrease
    # Make drift_ema high enough to trigger trust_radius decrease (drift_ema > 1.5 * trust_radius)
    controller = LyapunovController(kappa=0.1, r0=1.0)  # Reset controller for each case
    controller.trust_radius = 1.0
    controller.drift_ema = 1.6
    _, info = controller.update(d_star=1.0, grad_norm=1.0)
    assert np.isclose(info["trust_radius"], 1.0 * 0.8 - controller.integral_gain * 1.0)

    # 3. Test trust radius increase
    # Make drift_ema low enough to trigger trust_radius increase (drift_ema < 0.7 * trust_radius)
    controller = LyapunovController(kappa=0.1, r0=1.0)  # Reset controller for each case
    controller.trust_radius = 1.0
    controller.drift_ema = 0.6
    _, info = controller.update(d_star=1.0, grad_norm=1.0)
    assert np.isclose(info["trust_radius"], 1.0 * 1.1 - controller.integral_gain * 1.0)

    # 4. Test trust radius clipping (upper bound)
    controller = LyapunovController(kappa=0.1, r0=1.0)  # Reset controller for each case
    controller.trust_radius = 4.9
    controller.drift_ema = 0.1  # will increase trust_radius
    _, info = controller.update(d_star=1.0, grad_norm=1.0)
    assert np.isclose(info["trust_radius"], 5.0)  # 4.9 * 1.1 = 5.39, clipped to 5.0

    # 5. Test trust radius clipping (lower bound)
    controller = LyapunovController(kappa=0.1, r0=1.0)  # Reset controller for each case
    controller.trust_radius = 0.21
    controller.drift_ema = 2.0  # will decrease trust_radius
    _, info = controller.update(d_star=1.0, grad_norm=1.0)
    assert np.isclose(info["trust_radius"], 0.2)  # 0.21 * 0.8 = 0.168, clipped to 0.2

    # 6. Test neutral case (no change in trust_radius)
    controller = LyapunovController(kappa=0.1, r0=1.0)  # Reset controller for each case
    controller.trust_radius = 1.0
    controller.drift_ema = 1.0  # 0.7 <= 1.0 <= 1.5, so no change
    controller.drift_integral = 10.0
    _, info = controller.update(
        d_star=0.0, grad_norm=1.0
    )  # d_star=0 to not change drift_ema from 1.0
    assert np.isclose(controller.drift_integral, 9.5)


def test_lyapunov_controller_integral_adjustment_and_clipping_duplicate_removed() -> None:
    controller = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)

    # Test integral adjustment
    controller.drift_integral = 10.0
    controller.trust_radius = 1.0
    _, info = controller.update(d_star=0.0, grad_norm=1.0)  # d_star=0 to not change drift_ema
    # Expected: 1.0 - 0.01 * 10.0 = 0.9
    assert np.isclose(info["trust_radius"], 1.0)
    # Expected: 10.0 * 0.95 = 9.5
    assert np.isclose(controller.drift_integral, 9.5)

    # Test integral adjustment with clipping
    controller = LyapunovController(kappa=0.1, r0=0.25, integral_gain=0.01)
    controller.drift_integral = 10.0
    controller.trust_radius = 0.25
    _, info = controller.update(d_star=0.0, grad_norm=1.0)
    # Expected: 0.25 - 0.01 * 10.0 = 0.15, clipped to 0.2
    assert np.isclose(info["trust_radius"], 0.2)
    assert np.isclose(controller.drift_integral, 9.5)


def test_lyapunov_controller_integral_adjustment_and_clipping() -> None:
    controller = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)

    # Test integral adjustment
    controller.drift_integral = 10.0
    controller.trust_radius = 1.0
    _, info = controller.update(d_star=0.0, grad_norm=1.0)  # d_star=0 to not change drift_ema
    # Expected: 1.0 - 0.01 * 10.0 = 0.9
    assert np.isclose(info["trust_radius"], 1.0)
    # Expected: 10.0 * 0.95 = 9.5
    assert np.isclose(controller.drift_integral, 9.5)

    # Test integral adjustment with clipping
    controller = LyapunovController(kappa=0.1, r0=0.25, integral_gain=0.01)
    controller.drift_integral = 10.0
    controller.trust_radius = 0.25
    _, info = controller.update(d_star=0.0, grad_norm=1.0)
    # Expected: 0.25 - 0.01 * 10.0 = 0.15, clipped to 0.2
    assert np.isclose(info["trust_radius"], 0.2)
    assert np.isclose(controller.drift_integral, 9.5)


def test_ema_branch_exact_thresholds_are_neutral() -> None:
    """Existing tests exercise drift_ema clearly above/below the 1.5x/0.7x
    thresholds, never exactly at them -- a > vs >= (or < vs <=) mutation at
    either boundary would survive. integral_gain=0.0 isolates the EMA branch
    logic so trust_radius is only affected by whichever comparison fires."""
    upper = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.0)
    upper.drift_ema = 1.5
    upper.trust_radius = 1.0
    _, info = upper.update(d_star=1.5, grad_norm=1.0)  # drift_ema stays exactly 1.5
    assert np.isclose(info["drift_ema"], 1.5)
    assert np.isclose(info["trust_radius"], 1.0)  # exactly at 1.5x -> neither branch fires

    lower = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.0)
    lower.drift_ema = 0.7
    lower.trust_radius = 1.0
    _, info = lower.update(d_star=0.7, grad_norm=1.0)  # drift_ema stays exactly 0.7
    assert np.isclose(info["drift_ema"], 0.7)
    assert np.isclose(info["trust_radius"], 1.0)  # exactly at 0.7x -> neither branch fires
