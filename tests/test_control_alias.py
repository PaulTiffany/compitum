from __future__ import annotations

import numpy as np

from compitum.control import SRMFController


def test_srmf_alias_behaves_like_lyapunov() -> None:
    c = SRMFController(kappa=0.1, r0=1.0, integral_gain=0.01)
    # Single update
    eta, status = c.update(d_star=0.5, grad_norm=2.0)
    assert eta > 0.0
    assert 0.2 <= status["trust_radius"] <= 5.0

    # Batch update path
    d = np.array([0.2, 0.3, 0.4], float)
    g = np.array([1.0, 1.0, 1.0], float)
    etas, statuses = c.batch_update(d, g)
    assert len(etas) == 3 and len(statuses) == 3

