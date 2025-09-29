from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


class CalibratedPredictor:
    """
    Calibrated regressor with quantile bounds (p5,p95).
    For latency/cost: consider enabling monotonic constraints via LightGBM when available.
    """
    def __init__(self) -> None:
        self.base = GradientBoostingRegressor(random_state=42)
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.q05 = GradientBoostingRegressor(loss="quantile", alpha=0.05, random_state=41)
        self.q95 = GradientBoostingRegressor(loss="quantile", alpha=0.95, random_state=43)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.base.fit(X, y)
        raw = self.base.predict(X)
        self.iso.fit(raw, y)
        self.q05.fit(X, y)
        self.q95.fit(X, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = self.base.predict(X)
        y = self.iso.transform(raw)
        lo = self.q05.predict(X)
        hi = self.q95.predict(X)
        return y, lo, hi
