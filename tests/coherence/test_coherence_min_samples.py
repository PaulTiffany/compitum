import numpy as np

from compitum.coherence import CoherenceFunctional


def test_log_evidence_zero_until_min_samples():
    coh = CoherenceFunctional(k=100)
    rng = np.random.default_rng(0)
    d = 6
    # Fewer than 10 samples → no KDE
    for _ in range(9):
        coh.update("fast", rng.normal(size=d), success=1.0)
    v = coh.log_evidence("fast", np.zeros(d))
    assert v == 0.0
