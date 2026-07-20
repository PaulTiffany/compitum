import numpy as np
import pytest

from compitum.predictors import CalibratedPredictor


@pytest.mark.stat
def test_quantile_calibration_on_holdout():
    rng = np.random.default_rng(0)
    n, d = 300, 6
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (X @ w) + 0.2 * rng.standard_normal(n)

    Xtr, Xte = X[:200], X[200:]
    ytr, yte = y[:200], y[200:]

    cp = CalibratedPredictor()
    cp.fit(Xtr, ytr)
    y_hat, lo, hi = cp.predict(Xte)
    within = (y_hat >= lo) & (y_hat <= hi)
    assert float(np.mean(within)) >= 0.7
