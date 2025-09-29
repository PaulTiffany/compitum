from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KernelDensity


class WeightedReservoir:
    def __init__(self, k: int = 1000, rng: Optional[np.random.Generator] = None) -> None:
        self.k = k
        self.buf: List[Tuple[np.ndarray, float]] = []
        self.tot_w = 0.0
        self.rng = rng or np.random.default_rng()

    def add(self, x: np.ndarray, w: float) -> None:
        w = max(float(w), 1e-6)
        self.tot_w += w
        if len(self.buf) < self.k:
            self.buf.append((x.copy(), w))
        else:
            j = int(self.rng.integers(0, int(self.tot_w)))
            if j < self.k:
                self.buf[j] = (x.copy(), w)

class CoherenceFunctional:
    def __init__(self, k: int = 1000) -> None:
        self.res: defaultdict[str, WeightedReservoir] = defaultdict(lambda: WeightedReservoir(k))
        self.kde_cache: dict[str, KernelDensity] = {}

    def update(self, model_name: str, xw: np.ndarray, success: float) -> None:
        self.res[model_name].add(xw, success)
        self.kde_cache.pop(model_name, None)

    def _fit(self, model_name: str) -> KernelDensity | None:
        buf = self.res[model_name].buf
        if len(buf) < 10:
            return None
        X = np.stack([x for x, _ in buf], axis=0)
        w = np.array([wt for _, wt in buf], float)
        # Scott rule on whitened coords
        n, d = X.shape
        bw = n ** (-1.0 / (d + 4))
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(X, sample_weight=w / w.sum())
        self.kde_cache[model_name] = kde
        return kde

    # BRIDGEBLOCK_START def:coherence_log_evidence
    def log_evidence(self, model_name: str, xw: np.ndarray) -> float:
        kde = self.kde_cache.get(model_name) or self._fit(model_name)
        if kde is None:
            return 0.0
        val = float(kde.score_samples([xw])[0])
        return float(np.clip(val, -10.0, 10.0))
    # BRIDGEBLOCK_END def:coherence_log_evidence
