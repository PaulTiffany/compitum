from __future__ import annotations

import hashlib
from typing import Dict, Tuple, Union

import numpy as np


def split_features(x: Union[Dict[str, float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(x, dict):
        # Riemannian: everything except prag_*, Banach: prag_* only
        xR = [v for k, v in x.items() if not k.startswith("prag_")]
        xB = [v for k, v in x.items() if k.startswith("prag_")]
        return np.array(xR, float), np.array(xB, float)
    else:
        # Riemannian: everything except the last 4, Banach: last 4 only
        return x[:-4], x[-4:]


def pgd_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()
