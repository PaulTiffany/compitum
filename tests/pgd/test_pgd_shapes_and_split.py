import numpy as np

from compitum.pgd import RegexPromptExtractor
from compitum.utils import split_features


def test_pgd_feature_shapes_and_split():
    pgd = RegexPromptExtractor()
    x = pgd.extract_features("Prove AM-GM; 1, 2, 3.")
    # 35 Riemannian + 4 Banach
    assert isinstance(x, np.ndarray)
    assert x.shape[0] == 39

    xR, xB = split_features(x)
    assert xR.shape[0] == 35
    assert xB.shape[0] == 4
