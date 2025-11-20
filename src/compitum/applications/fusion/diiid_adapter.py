from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ShotData:
    time_ms: np.ndarray
    state: np.ndarray  # shape (T, D)
    q_min: np.ndarray  # shape (T,)
    crash_index: Optional[int]


REQUIRED_COLS = ("time_ms", "q_min")


def load_shot_csv(path: Path, state_dim: int = 8) -> ShotData:
    """
    Minimal CSV loader for offline evaluation.

    Expected columns (at minimum):
      - time_ms: float
      - q_min: float
    Optional columns used if present:
      - Te_core, ne (and others). Missing dims are zero-filled.

    The state vector layout is assumed to be:
      [Te_core, ne, q_min, x3, x4, ..., x{D-1}] (zeros if missing)
    """
    df = pd.read_csv(path)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    T = len(df)
    D = int(state_dim)
    state = np.zeros((T, D), dtype=float)

    # Map known columns
    if "Te_core" in df.columns:
        state[:, 0] = df["Te_core"].to_numpy(dtype=float)
    if "ne" in df.columns:
        state[:, 1] = df["ne"].to_numpy(dtype=float)
    # q_min at index 2 by convention (matches demo)
    state[:, 2] = df["q_min"].to_numpy(dtype=float)

    t = df["time_ms"].to_numpy(dtype=float)
    q = state[:, 2].copy()

    # First time q_min drops below 1.0 is considered a crash (if any)
    below = np.nonzero(q < 1.0)[0]
    crash_idx: Optional[int] = int(below[0]) if len(below) else None

    return ShotData(time_ms=t, state=state, q_min=q, crash_index=crash_idx)
