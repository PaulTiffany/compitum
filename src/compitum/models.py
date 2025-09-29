from dataclasses import dataclass

import numpy as np

from .capabilities import Capabilities


@dataclass
class Model:
    name: str
    center: np.ndarray  # center in Riemannian feature space
    capabilities: Capabilities
    cost: float
