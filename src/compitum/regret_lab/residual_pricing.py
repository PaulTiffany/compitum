"""Bounded, gated residual-price adapter (tranche 5).

Wraps a frozen ``PacingController`` (tranches 4/4.6, completely
unmodified) and adds a bounded, gated correction on top:

    lambda_effective[t] = clip(lambda_base[t] + gate[t] * delta_lambda_pc[t],
                                0, lambda_max)

Every step's correction attempt is recorded as a ``ResidualCorrectionRecord``
(status one of ``applied``/``zero_gate``/``clipped``/``refused``/``failed``)
-- a lightweight, inspectable provenance trail, analogous in spirit to
tranches 1-3's governed ``TrajectoryEvidence`` status vocabulary without
depending on that module's FabricPC-observation-specific raw schema. Any
predictor exception or refusal degrades deterministically to a zero
correction -- never to a crash, never to a stale/partial correction. See
docs/adr/0006-fabricpc-residual-shadow-pricing.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .pricing import PacingController, PricingUpdateContext
from .residual_channels import (
    ResidualChannelHistory,
    advance_history,
    compute_residual_channel_vector,
)

ResidualPredictor = Callable[[List[np.ndarray]], Optional[float]]
GateFn = Callable[[PricingUpdateContext], bool]


def _gate_always_open(context: PricingUpdateContext) -> bool:
    return True


@dataclass
class ResidualCorrectionRecord:
    step: int
    status: str
    raw_correction: Optional[float]
    applied_correction: float
    window_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status,
            "raw_correction": self.raw_correction,
            "applied_correction": self.applied_correction,
            "window_size": self.window_size,
        }


@dataclass
class ResidualPricingController:
    base: PacingController
    predict_residual: ResidualPredictor
    max_correction_magnitude: float
    gate_fn: GateFn = _gate_always_open
    window_size: int = 5
    lambda_max: float = 20.0
    _window: List[np.ndarray] = field(default_factory=list, repr=False)
    _pending_correction: float = field(default=0.0, repr=False)
    _history: ResidualChannelHistory = field(default_factory=ResidualChannelHistory, repr=False)
    records: List[ResidualCorrectionRecord] = field(default_factory=list, repr=False)

    @property
    def lambda_price(self) -> Dict[str, float]:
        base_price = self.base.lambda_price
        return {
            r: min(max(v + self._pending_correction, 0.0), self.lambda_max)
            for r, v in base_price.items()
        }

    def update(self, context: PricingUpdateContext) -> None:
        self.base.update(context)

        if context.case is None or context.chosen is None:
            # Structurally required for this controller; a caller that
            # omits them is misconfigured, not a governed runtime failure.
            raise ValueError(
                "ResidualPricingController requires PricingUpdateContext.case "
                "and .chosen -- pass them from simulate_policy"
            )

        resource = context.resource_names[0]
        lambda_base = self.base.lambda_price.get(resource, 0.0)
        # The base controller's own cumulative-usage error isn't exposed
        # publicly; its lambda_price already reflects that error (a
        # nonzero price only exists because the error pushed it there), so
        # it stands in directly as "how consequential does pacing currently
        # think scarcity is".
        pacing_error = lambda_base

        vector = compute_residual_channel_vector(
            remaining=context.remaining_before.get(resource, 0.0),
            case=context.case,
            lambda_base=lambda_base,
            pacing_error=pacing_error,
            history=self._history,
            steps_left=context.total_steps - context.step,
            total_steps=context.total_steps,
            resource=resource,
        )
        self._window.append(vector)
        if len(self._window) > self.window_size:
            self._window.pop(0)

        self._history = advance_history(self._history, context.case, context.chosen, lambda_base)

        if not self.gate_fn(context):
            self._pending_correction = 0.0
            self.records.append(
                ResidualCorrectionRecord(
                    step=context.step,
                    status="zero_gate",
                    raw_correction=None,
                    applied_correction=0.0,
                    window_size=len(self._window),
                )
            )
            return

        try:
            raw = self.predict_residual(list(self._window))
        except Exception:
            raw = None

        if raw is None:
            self._pending_correction = 0.0
            self.records.append(
                ResidualCorrectionRecord(
                    step=context.step,
                    status="failed",
                    raw_correction=None,
                    applied_correction=0.0,
                    window_size=len(self._window),
                )
            )
            return

        clipped = max(-self.max_correction_magnitude, min(self.max_correction_magnitude, raw))
        status = "applied" if clipped == raw else "clipped"
        self._pending_correction = clipped
        self.records.append(
            ResidualCorrectionRecord(
                step=context.step,
                status=status,
                raw_correction=raw,
                applied_correction=clipped,
                window_size=len(self._window),
            )
        )
