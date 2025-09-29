import numpy as np

from compitum.predictors import CalibratedPredictor


def test_calibrated_predictor() -> None:
    predictor = CalibratedPredictor()
    assert not predictor.fitted

    # Create synthetic data
    rng = np.random.default_rng(42)
    X_train = rng.random((100, 5))
    y_train = rng.random(100)

    # Fit the model
    predictor.fit(X_train, y_train)
    assert predictor.fitted

    # Predict
    X_test = rng.random((10, 5))
    y, lo, hi = predictor.predict(X_test)

    # Check output shapes
    assert y.shape == (10,)
    assert lo.shape == (10,)
    assert hi.shape == (10,)

    # Check that the lower bound is less than or equal to the upper bound.
    assert np.all(lo <= hi)
