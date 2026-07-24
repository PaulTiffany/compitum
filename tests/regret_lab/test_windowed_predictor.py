"""Generic ridge predictor + window flattening -- verified against a
genuinely learnable synthetic relationship, not just "runs"."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.regret_lab.windowed_predictor import fit_ridge, flatten_window, predict_ridge


def test_ridge_recovers_a_known_linear_relationship() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    true_weights = np.array([2.0, -1.0, 0.5])
    y = X @ true_weights + 1.0

    model = fit_ridge(X, y)
    X_test = rng.normal(size=(50, 3))
    y_test = X_test @ true_weights + 1.0
    predictions = [predict_ridge(model, row) for row in X_test]
    mae = np.mean(np.abs(np.array(predictions) - y_test))
    assert mae < 0.5


def test_ridge_constant_target_predicts_the_constant() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([3.0, 3.0, 3.0, 3.0])
    model = fit_ridge(X, y)
    assert predict_ridge(model, [1.0, 2.0]) == pytest.approx(3.0)


def test_ridge_handles_zero_variance_feature() -> None:
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = fit_ridge(X, y)
    assert model.feature_scale[1] == 1.0  # zero-variance column guarded to 1.0
    prediction = predict_ridge(model, [5.0, 5.0])
    assert np.isfinite(prediction)


def test_flatten_window_pads_short_windows_with_zeros() -> None:
    window = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    flattened = flatten_window(window, max_window=3, channel_dim=2)
    assert flattened.shape == (6,)
    assert list(flattened) == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]


def test_flatten_window_truncates_to_the_most_recent_steps() -> None:
    window = [np.array([float(i)]) for i in range(5)]
    flattened = flatten_window(window, max_window=2, channel_dim=1)
    assert list(flattened) == [3.0, 4.0]


def test_flatten_window_exact_size_needs_no_padding_or_truncation() -> None:
    window = [np.array([1.0]), np.array([2.0])]
    flattened = flatten_window(window, max_window=2, channel_dim=1)
    assert list(flattened) == [1.0, 2.0]
