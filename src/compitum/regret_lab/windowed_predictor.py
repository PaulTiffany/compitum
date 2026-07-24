"""Generic ridge-regression predictor over a flattened window (tranche 5).

Used two ways: (1) directly on the flattened declared window as the
non-FabricPC windowed comparator (arm 3); (2) on FabricPC-derived
trajectory-summary features as the offline fit mapping those features to a
residual prediction (arms 4-7). Both are fit offline on training sequences
and frozen at test time, exactly like tranches 2/3's ridge/EWMA baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

RIDGE_LAMBDA = 1.0


@dataclass
class RidgeModel:
    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray


def fit_ridge(
    features: Sequence[Sequence[float]], targets: Sequence[float], ridge: float = RIDGE_LAMBDA
) -> RidgeModel:
    X = np.array(features, dtype=float)
    y = np.array(targets, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0.0] = 1.0
    Xn = (X - mean) / scale
    y_mean = float(y.mean())
    gram = Xn.T @ Xn + ridge * np.eye(Xn.shape[1])
    weights = np.linalg.solve(gram, Xn.T @ (y - y_mean))
    return RidgeModel(weights=weights, bias=y_mean, feature_mean=mean, feature_scale=scale)


def predict_ridge(model: RidgeModel, features: Sequence[float]) -> float:
    x = np.array(features, dtype=float)
    xn = (x - model.feature_mean) / model.feature_scale
    return float(xn @ model.weights + model.bias)


def flatten_window(window: List[np.ndarray], max_window: int, channel_dim: int) -> np.ndarray:
    """Zero-pads (at the front, i.e. the "past" end) so the flattened
    vector always has dimension ``max_window * channel_dim`` regardless of
    how early in the sequence the window was taken -- a fixed-size input a
    ridge model (or FabricPC's source layer) can always accept."""
    padding_needed = max_window - len(window)
    if padding_needed > 0:
        pad = [np.zeros(channel_dim) for _ in range(padding_needed)]
        full = pad + list(window)
    else:
        full = list(window[-max_window:])
    return np.concatenate(full)
