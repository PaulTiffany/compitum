from __future__ import annotations

from typing import Any, Dict

import numpy as np


class BoundaryAnalyzer:
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
        is_boundary = (gap < 0.05 or entropy > 0.65) and (sigma > 0.12)
        # BRIDGEBLOCK_END def:boundary_condition
        return {"winner": m1, "runner_up": m2, "utility_gap": float(gap),
                "entropy": float(entropy), "uncertainty": sigma, "is_boundary": bool(is_boundary)}
