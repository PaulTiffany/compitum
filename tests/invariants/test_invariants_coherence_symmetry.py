import numpy as np

from compitum.coherence import CoherenceFunctional


def test_isotropic_symmetry_evidence_matches_for_plus_minus():
    coh = CoherenceFunctional(k=1000)
    rng = np.random.default_rng(0)
    d = 8
    for _ in range(400):
        x = rng.normal(0.0, 0.5, size=d)
        coh.update("fast", x, success=1.0)

    v = rng.normal(0.0, 1.0, size=d)
    v /= np.linalg.norm(v) + 1e-9
    val_pos = coh.log_evidence("fast", 0.8 * v)
    val_neg = coh.log_evidence("fast", -0.8 * v)
    # Isotropic cloud → symmetry implies evidence should be close for +/- v
    assert abs(val_pos - val_neg) <= 0.05
