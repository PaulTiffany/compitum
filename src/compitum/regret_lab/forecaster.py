"""Simple non-FabricPC sequential consumption forecaster (tranche 3, arm 3).

Stands in for "a simple sequential non-FabricPC model" the FabricPC arm
must also beat, not just the no-predictor dual baseline. Tracks an
exponentially-weighted moving average of each (model, resource)'s
forecast residual (``realized - expected``) and applies it as a correction
to future forecasts -- enough to help when a scenario's forecast error is
systematically biased (``forecast_error``), and, correctly, not much when
the error is closer to unbiased noise (``delayed_realization``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class EWMAForecaster:
    alpha: float = 0.3
    _bias: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def predict(
        self, expected_consumption: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        predicted: Dict[str, Dict[str, float]] = {}
        for model, resources in expected_consumption.items():
            predicted[model] = {
                r: v + self._bias.get((model, r), 0.0) for r, v in resources.items()
            }
        return predicted

    def update(
        self,
        chosen_model: str,
        expected_consumption: Dict[str, float],
        realized_consumption: Dict[str, float],
    ) -> None:
        """Correct only the *chosen* model's bias -- a real online policy only
        ever observes the realized outcome of the route it actually took,
        never a would-be outcome for routes it didn't select."""
        for r, realized_value in realized_consumption.items():
            expected_value = expected_consumption[r]
            residual = realized_value - expected_value
            key = (chosen_model, r)
            previous = self._bias.get(key, 0.0)
            self._bias[key] = (1.0 - self.alpha) * previous + self.alpha * residual

    def __call__(
        self, expected_consumption: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        return self.predict(expected_consumption)
