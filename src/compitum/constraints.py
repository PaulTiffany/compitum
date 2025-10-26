from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np

from .models import Model


class InfoDict(TypedDict):
    status: str
    violations: List[str]
    feasible: bool
    shadow_prices: Dict[str, float]


class ReflectiveConstraintSolver:
    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self.A, self.b = A, b
        self.last_viable_models: List[Any] = []

    def _is_feasible(
        self, model: "Model", xB: np.ndarray, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Checks if a model is feasible given the PGD vector and optional context."""
        if context is None:
            ok = model.capabilities.supports(xB)
        else:
            ok = model.capabilities.supports(xB, context=context)
        if not ok:
            return False
        if not np.all(self.A @ xB <= self.b + 1e-10):
            return False
        return True

    def select(
        self,
        xB: np.ndarray,
        models: List[Model],
        utilities: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Model, InfoDict]:
        sorted_models = sorted(models, key=lambda m: utilities.get(m.name, -np.inf), reverse=True)

        viable = []
        for model in sorted_models:
            if self._is_feasible(model, xB, context=context):
                viable.append(model)

        info: InfoDict
        if not viable:
            info = {
                "status": "infeasible_fallback",
                "violations": ["all_models_violate_constraints"],
                "feasible": False,
                "shadow_prices": {},
            }
            for i in range(len(self.b)):
                info["shadow_prices"][f"lambda_{i}"] = 0.0
            return sorted_models[0], info

        m_star = viable[0]
        info = {
            "status": "optimal",
            "violations": [],
            "feasible": True,
            "shadow_prices": {},
        }
        # Shadow price calculation
        for i in range(len(self.b)):
            b_relaxed = self.b.copy()
            b_relaxed[i] += 1e-5

            shadow_price = 0.0
            for competitor in sorted_models:
                if competitor == m_star:
                    continue

                # Check if competitor is viable under relaxed constraints
                is_competitor_viable_relaxed = True
                if context is None:
                    ok_cap = competitor.capabilities.supports(xB)
                else:
                    ok_cap = competitor.capabilities.supports(xB, context=context)
                if not ok_cap:
                    is_competitor_viable_relaxed = False

                # Create a temporary solver with the relaxed constraints
                temp_solver = ReflectiveConstraintSolver(self.A, b_relaxed)
                if not temp_solver._is_feasible(competitor, xB, context=context):
                    is_competitor_viable_relaxed = False

                if is_competitor_viable_relaxed:
                    utility_competitor = utilities.get(competitor.name, -np.inf)
                    utility_m_star = utilities.get(m_star.name, -np.inf)
                    if utility_competitor > utility_m_star:
                        shadow_price = (utility_competitor - utility_m_star) / 1e-5
                        break
            info["shadow_prices"][f"lambda_{i}"] = shadow_price

        return m_star, info
