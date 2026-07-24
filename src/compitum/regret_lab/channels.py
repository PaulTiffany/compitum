"""Declared FabricPC input channel for the dynamic-regret environment
(tranche 3). Dependency-free: FabricPC observes only state the online
simulator already has at decision time -- never a future or realized
outcome. Mirrors tranche 2's declared-channel-mapping discipline
(``compitum.constraint_oracle.channels``): the mapping is fixed, ordered,
and documented, not left implicit in the JAX-side glue.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .environment import MODEL_NAMES, RESOURCE_NAMES, DynamicCase

CHANNEL_DIMENSION = 15
_BUDGET_NORM = 10.0


def compute_regret_channel_vector(
    remaining: Dict[str, float],
    case: DynamicCase,
    lambda_price: Dict[str, float],
    steps_left: int,
    total_steps: int,
) -> np.ndarray:
    """Fixed 15-dim vector, in declared order:

    ``[0:2]``   remaining budget/quota, normalized by ``_BUDGET_NORM``
    ``[2:8]``   forecast-available expected consumption, model x resource
                (``MODEL_NAMES x RESOURCE_NAMES`` order)
    ``[8:11]``  base utility per model
    ``[11:13]`` current dual price per resource
    ``[13]``    fraction of the sequence remaining
    ``[14]``    total replenishment this step (a simple resource-inflow signal)
    """
    vector = np.zeros(CHANNEL_DIMENSION, dtype=float)
    for i, r in enumerate(RESOURCE_NAMES):
        vector[i] = remaining.get(r, 0.0) / _BUDGET_NORM

    idx = 2
    for m in MODEL_NAMES:
        for r in RESOURCE_NAMES:
            vector[idx] = case.expected_consumption[m][r]
            idx += 1

    for i, m in enumerate(MODEL_NAMES):
        vector[8 + i] = case.base_utility.get(m, 0.0)

    for i, r in enumerate(RESOURCE_NAMES):
        vector[11 + i] = lambda_price.get(r, 0.0)

    vector[13] = steps_left / total_steps if total_steps > 0 else 0.0
    vector[14] = sum(case.replenishment.values())
    return vector
