import numpy as np

from compitum.utils import split_features


def test_split_features_from_array_tail():
    x = np.arange(10, dtype=float)
    R, B = split_features(x)
    assert np.allclose(B, x[-4:])
    assert np.allclose(R, x[:-4])


def test_split_features_from_dict_pragmatic_keys():
    feats = {
        "a": 1.0,
        "b": 2.0,
        "prag_latency_class": 3.0,
        "prag_cost_class": 4.0,
        "c": 5.0,
        "prag_pii_level": 6.0,
        "prag_region_eu_only": 7.0,
    }
    R, B = split_features(feats)
    # Banach vector should contain only prag_* in insertion order
    assert np.allclose(B, [3.0, 4.0, 6.0, 7.0])
    # Riemannian vector should contain the rest in insertion order
    assert np.allclose(R, [1.0, 2.0, 5.0])
