import numpy as np
from hypothesis import given, strategies as st

from compitum.predictors import CalibratedPredictor


@given(n=st.integers(min_value=50, max_value=200), d=st.integers(min_value=2, max_value=10))
def test_quantile_bounds_envelope(n: int, d: int) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (X @ w) + 0.1 * rng.standard_normal(n)  # noisy linear

    cp = CalibratedPredictor()
    cp.fit(X, y)
    y_hat, lo, hi = cp.predict(X)

    # On training data, isotonic-calibrated output should be within quantile envelope most of the time
    within = (y_hat >= lo) & (y_hat <= hi)
    # Allow small violations due to model mismatch; expect majority to be within
    assert float(np.mean(within)) >= 0.8
