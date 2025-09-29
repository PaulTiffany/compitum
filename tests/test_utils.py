import numpy as np

from compitum.utils import split_features


def test_split_features() -> None:
    features = {
        "prag_a": 1.0,
        "prag_b": 2.0,
        "riem_c": 3.0,
        "riem_d": 4.0,
    }
    xR, xB = split_features(features)
    np.testing.assert_array_equal(xR, np.array([3.0, 4.0]))
    np.testing.assert_array_equal(xB, np.array([1.0, 2.0]))


