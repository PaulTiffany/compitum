from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np


def split_features(x: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    # Riemannian: everything except prag_*, Banach: prag_* only
    xR = [v for k, v in x.items() if not k.startswith("prag_")]
    xB = [v for k, v in x.items() if k.startswith("prag_")]
    return np.array(xR, float), np.array(xB, float)

def pgd_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()
