from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .models import Model


class ReflectiveConstraintSolver:
    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self.A, self.b = A, b
        self.last_viable_models: List[Any] = []

    def _is_feasible(self, model: Model, pgd_banach: np.ndarray) -> bool:
        if not np.all(self.A @ pgd_banach <= self.b + 1e-10):
            return False
        return model.capabilities.supports(pgd_banach)

    def select(self, pgd_banach: np.ndarray, models: List[Model],
               utilities: Dict[str, float], eps: float = 1e-3) -> Tuple[Model, Dict[str, Any]]:
        viable = [m for m in models if self._is_feasible(m, pgd_banach)]
        self.last_viable_models = viable
        if not viable:
            m_star = max(models, key=lambda m: utilities[m.name])
            return m_star, {"feasible": False, "minimal_violation": True,
                            "binding_constraints": [], "shadow_prices": {}}

        m_star = max(viable, key=lambda m: utilities[m.name])

        # BRIDGEBLOCK_START alg:shadow_price_calculation
        lambdas: Dict[str, float] = {}
        for j in range(self.b.size):
            b_relaxed = self.b.copy()
            b_relaxed[j] += eps
            # if relaxation changes feasibility of better competitors, estimate ∂U/∂b_j
            best_util = utilities[m_star.name]
            for comp in models:
                if comp in viable or utilities[comp.name] <= best_util:
                    continue
                ok = (
                    np.all(self.A @ pgd_banach <= b_relaxed + 1e-10) and
                    comp.capabilities.supports(pgd_banach)
                )
                if ok:
                    best_util = max(best_util, utilities[comp.name])
            lambdas[f"lambda_{j}"] = max(0.0, (best_util - utilities[m_star.name]) / eps)
        # BRIDGEBLOCK_END alg:shadow_price_calculation

        binding = [j for j, val in enumerate(self.A @ pgd_banach) if val >= self.b[j] - 1e-9]
        return m_star, {"feasible": True, "minimal_violation": False,
                        "binding_constraints": binding, "shadow_prices": lambdas}
