import numpy as np

from compitum.control import LyapunovController


def test_lyapunov_nonincrease_over_zero_drift_sequence():
    ctrl = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
    # Warm start to a nonzero state
    ctrl.update(d_star=0.4, grad_norm=1.0)
    v0 = ctrl.lyapunov_function()
    # Multiple steps with zero drift should not increase V
    for _ in range(50):
        ctrl.update(d_star=0.0, grad_norm=1.0)
    vT = ctrl.lyapunov_function()
    assert vT <= v0 + 1e-12


def test_lyapunov_final_below_initial_under_decreasing_drift():
    ctrl = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
    # Start with moderate drift for a few steps
    for _ in range(10):
        ctrl.update(d_star=0.5, grad_norm=1.0)
    v_init = ctrl.lyapunov_function()

    # Then feed a decreasing drift schedule towards zero
    for d in np.linspace(0.3, 0.0, 40):
        ctrl.update(d_star=float(d), grad_norm=1.0)
    v_final = ctrl.lyapunov_function()
    # Expect net recovery: final Lyapunov proxy below initial
    assert v_final < v_init

