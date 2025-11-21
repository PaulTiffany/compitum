from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from compitum.control import LyapunovController


@dataclass
class SRMFState:
    drift: float
    constraint: float
    bias: float

    def current_phase(self) -> str:
        if self.drift > self.constraint and self.drift > self.bias:
            return "drift"
        if self.constraint > self.drift and self.constraint > self.bias:
            return "constraint"
        return "bias"


def map_material_to_srmf(doc: Any) -> SRMFState:
    """Map a Materials Project-like summary doc to SRMF proxy coordinates.

    Robust to None/missing attributes; uses simple proxies:
    - drift ~ 1/(band_gap + eps)
    - constraint ~ density * log(nsites)
    - bias ~ |formation_energy_per_atom|
    """
    band_gap = getattr(doc, "band_gap", None) or 0.0
    density = getattr(doc, "density", None) or 0.0
    nsites = getattr(doc, "nsites", None) or 1
    fe_pa = getattr(doc, "formation_energy_per_atom", None)
    fe_pa = 0.0 if fe_pa is None else float(fe_pa)

    drift = 1.0 / (float(band_gap) + 0.01)
    constraint = float(density) * float(np.log(max(int(nsites), 1)))
    bias = abs(fe_pa)
    return SRMFState(drift=drift, constraint=constraint, bias=bias)


def _curvature_kappa(state: SRMFState) -> float:
    """Lightweight curvature proxy capturing drift vs. stabilizers."""
    denom = 1.0 + state.constraint + state.bias
    return float(state.drift / denom)


def _lyapunov_leak(state: SRMFState) -> float:
    """Lyapunov leak proxy using Compitum's controller on drift."""
    ctrl = LyapunovController(kappa=0.1, r0=1.0, integral_gain=0.01)
    _, status = ctrl.update(d_star=float(state.drift), grad_norm=1.0)
    return float(status.get("lyapunov_function", 0.0))


def audit_the_manifold(
    api_key: str,
    search_criteria: Dict[str, Any],
    *,
    kappa_threshold: float = 0.5,
    leak_threshold: float = 0.1,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Query Materials Project and compute SRMF/curvature/leak for each hit.

    Returns a DataFrame with columns:
    material_id, formula, srmf_phase, curvature_kappa, stability_leak, prediction
    """
    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "mp_api is required. Install mp_api or use extras materials."
        ) from e

    _fields = fields or [
        "material_id",
        "formula_pretty",
        "band_gap",
        "density",
        "nsites",
        "formation_energy_per_atom",
    ]

    with MPRester(api_key) as mpr:  # pragma: no cover (exercised in notebook)
        docs: Iterable[Any] = mpr.materials.summary.search(**search_criteria, fields=_fields)

    rows: list[dict[str, Any]] = []
    for doc in docs:
        state = map_material_to_srmf(doc)
        kappa = _curvature_kappa(state)
        leak = _lyapunov_leak(state)
        pred = (
            'candidate' if (kappa >= float(kappa_threshold) and leak <= float(leak_threshold)) else 'non_candidate'
        )
        rows.append(
            dict(
                material_id=getattr(doc, "material_id", ""),
                formula=getattr(doc, "formula_pretty", ""),
                srmf_phase=state.current_phase(),
                curvature_kappa=float(kappa),
                stability_leak=float(leak),
                prediction=pred,
            )
        )

    return pd.DataFrame(rows)






