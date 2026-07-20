import numpy as np

from compitum.coherence import CoherenceFunctional


def test_log_evidence_higher_near_training_cluster():
    coh = CoherenceFunctional(k=100)
    rng = np.random.default_rng(0)
    # Generate 20 points near origin in 4D whitened space
    for _ in range(20):
        xw = rng.normal(0.0, 0.2, size=4)
        coh.update("m1", xw.astype(float), success=1.0)

    near = np.zeros(4)
    far = np.full(4, 5.0)
    # After enough points, KDE should be available and near > far
    e_near = coh.log_evidence("m1", near)
    e_far = coh.log_evidence("m1", far)
    assert e_near > e_far
    # Bounded
    assert -10.0 <= e_near <= 10.0
