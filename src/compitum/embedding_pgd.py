from __future__ import annotations

from typing import Dict

import numpy as np


class EmbeddingPGDExtractor:
    """
    A PGD extractor that works with pre-computed embeddings.
    """

    def __init__(self) -> None:
        pass

    def extract_features(self, embedding: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns the embedding as a dictionary of features.
        """
        return {"embedding": embedding}
