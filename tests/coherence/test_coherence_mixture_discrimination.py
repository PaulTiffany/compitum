import numpy as np
import pytest

from compitum.coherence import CoherenceFunctional


@pytest.mark.lg
def test_mixture_discrimination_own_center_wins():
    rng = np.random.default_rng(0)
    coh = CoherenceFunctional(k=2000)
    d = 8
    mu_a = np.zeros(d)
    mu_b = np.zeros(d)
    mu_b[0] = 2.0
    # Populate two clusters for different model names
    for _ in range(600):
        coh.update("A", rng.normal(mu_a, 0.4), success=1.0)
    for _ in range(600):
        coh.update("B", rng.normal(mu_b, 0.4), success=1.0)

    # Near mu_a, A should have higher log evidence than B
    x = mu_a + rng.normal(0.0, 0.1, size=d)
    la = coh.log_evidence("A", x)
    lb = coh.log_evidence("B", x)
    assert la >= lb - 0.05

    # Near mu_b, B should have higher log evidence than A
    x = mu_b + rng.normal(0.0, 0.1, size=d)
    la2 = coh.log_evidence("A", x)
    lb2 = coh.log_evidence("B", x)
    assert lb2 >= la2 - 0.05
