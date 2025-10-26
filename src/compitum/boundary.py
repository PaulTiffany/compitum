from __future__ import annotations

from typing import Any, Dict

import numpy as np


class BoundaryAnalyzer:
    def __init__(
        self,
        gap_threshold: float = 0.05,
        entropy_threshold: float = 0.65,
        sigma_threshold: float = 0.12,
    ) -> None:
        self.gap_threshold = gap_threshold
        self.entropy_threshold = entropy_threshold
        self.sigma_threshold = sigma_threshold

    def analyze(self, utilities: Dict[str, float], u_sigma: Dict[str, float]) -> Dict[str, Any]:
        if len(utilities) < 2:
            return {"is_boundary": False, "reason": "insufficient_models"}
        items = sorted(utilities.items(), key=lambda kv: kv[1], reverse=True)
        (m1, u1), (m2, u2) = items[0], items[1]
        gap = u1 - u2
        arr = np.array([u for _, u in items])
        probs = np.exp(arr - u1)
        probs /= probs.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        sigma = float(u_sigma.get(m1, 0.0))
        # BRIDGEBLOCK_START def:boundary_condition
        is_boundary = (
            (gap < self.gap_threshold or entropy > self.entropy_threshold)
            and (sigma > self.sigma_threshold)
        )
        # BRIDGEBLOCK_END def:boundary_condition
        return {
            "winner": m1,
            "runner_up": m2,
            "utility_gap": float(gap),
            "entropy": float(entropy),
            "uncertainty": sigma,
            "is_boundary": bool(is_boundary),
        }
