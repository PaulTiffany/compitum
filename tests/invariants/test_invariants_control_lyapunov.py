import numpy as np

from compitum.control import LyapunovController


def test_lyapunov_decreases_with_zero_drift():
    ctrl = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
    # Warm up once to establish nonzero state
    ctrl.update(d_star=0.3, grad_norm=1.0)
    v_prev = ctrl.lyapunov_function()
    # With zero drift, the proxy should decay monotonically under the built-in decay
    for _ in range(10):
        ctrl.update(d_star=0.0, grad_norm=1.0)
        v_now = ctrl.lyapunov_function()
        assert v_now <= v_prev + 1e-12
        v_prev = v_now


def test_trust_radius_saturates_under_sustained_drift():
    ctrl = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.0)
    # High persistent drift; controller should not diverge and will saturate at the upper bound
    for _ in range(60):
        ctrl.update(d_star=2.0, grad_norm=1.0)
    assert 0.2 <= ctrl.trust_radius <= 5.0
    assert abs(ctrl.trust_radius - 5.0) < 1e-9


def test_lyapunov_recovers_after_drift_stops():
    ctrl = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.005)
    # Drive state with sustained drift
    for _ in range(30):
        ctrl.update(d_star=2.0, grad_norm=1.0)
    # Then zero drift; Lyapunov proxy should decrease over time
    v_prev = ctrl.lyapunov_function()
    decreased = False
    for _ in range(30):
        ctrl.update(d_star=0.0, grad_norm=1.0)
        v_now = ctrl.lyapunov_function()
        if v_now < v_prev - 1e-12:
            decreased = True
            break
        v_prev = v_now
    assert decreased
