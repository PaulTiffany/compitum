"""Experiment-only online primal-dual reference controller (tranche 3).

Established *before* any FabricPC involvement, per the governing
instruction that FabricPC needs a valid dual baseline to improve upon, not
a strawman. Deliberately not claimed to be optimal or production-ready --
it is the minimum coherent baseline every other arm (and FabricPC) is
compared against. Kept semantically and structurally separate from
``constraints.shadow_prices``: different module, different field name
(``lambda_price`` here vs. ``shadow_prices`` in the frozen solver), never
assigned into or read from the production path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class DualController:
    resource_names: Tuple[str, ...]
    eta: float = 0.5
    lambda_max: float = 20.0
    lambda_price: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lambda_price:
            self.lambda_price = {r: 0.0 for r in self.resource_names}

    def update(self, utilization_error: Dict[str, float]) -> None:
        """``lambda[t+1] = clip(lambda[t] + eta * utilization_error, 0, lambda_max)``."""
        for r in self.resource_names:
            updated = self.lambda_price[r] + self.eta * utilization_error.get(r, 0.0)
            self.lambda_price[r] = min(max(updated, 0.0), self.lambda_max)


def price_utilities(
    base_utility: Dict[str, float],
    expected_consumption: Dict[str, Dict[str, float]],
    lambda_price: Dict[str, float],
) -> Dict[str, float]:
    """``priced_utility[model] = base_utility[model]
    - dot(lambda, expected_consumption[model])``."""
    priced = {}
    for model, utility in base_utility.items():
        cost = sum(
            lambda_price.get(r, 0.0) * expected_consumption[model].get(r, 0.0) for r in lambda_price
        )
        priced[model] = utility - cost
    return priced
