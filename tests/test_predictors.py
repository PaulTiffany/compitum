import numpy as np

from compitum.predictors import CalibratedPredictor


def test_calibrated_predictor() -> None:
    predictor = CalibratedPredictor()
    # `not predictor.fitted` passes for both `False` and `None` -- use strict
    # identity so a mutated `fitted = None` default doesn't survive.
    assert predictor.fitted is False

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


def test_calibrated_predictor_hyperparameters_are_exact() -> None:
    """None of the shape/envelope-style tests above distinguish between
    `n_estimators=5` and `=6`, or between the three regressors' distinct
    `random_state` seeds -- with limited synthetic data, one extra boosting
    round or a different seed rarely changes predictions enough to trip a
    loose behavioral assertion, and pinning that behaviorally would be
    fragile/expensive. These are directly observable via sklearn's public
    constructor attributes without needing to fit anything."""
    predictor = CalibratedPredictor()
    assert predictor.base.n_estimators == 5
    assert predictor.base.random_state == 42
    assert predictor.q05.n_estimators == 5
    assert predictor.q05.random_state == 41
    assert predictor.q95.n_estimators == 5
    assert predictor.q95.random_state == 43


def test_isotonic_calibration_clips_out_of_domain_raw_values() -> None:
    """out_of_bounds="clip" is never exercised by shape/envelope-style tests,
    since typical test inputs stay within the base model's observed raw-value
    range. Directly verify that raw predictions outside the fitted domain clip
    to the boundary's fitted y, rather than extrapolating."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 3))
    w = np.array([1.0, 0.5, -0.5])
    y_train = X_train @ w

    predictor = CalibratedPredictor()
    predictor.fit(X_train, y_train)

    raw_train = predictor.base.predict(X_train)
    y_at_min = predictor.iso.transform(np.array([raw_train.min()]))
    y_at_max = predictor.iso.transform(np.array([raw_train.max()]))

    far_below = predictor.iso.transform(np.array([raw_train.min() - 100.0]))
    far_above = predictor.iso.transform(np.array([raw_train.max() + 100.0]))

    assert np.isclose(far_below, y_at_min)
    assert np.isclose(far_above, y_at_max)
