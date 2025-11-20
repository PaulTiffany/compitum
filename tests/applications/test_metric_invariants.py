import math
from typing import List

import numpy as np
from hypothesis import given, strategies as st, settings

from compitum.metric import SymbolicManifoldMetric


@st.composite
def vec(draw, dim: int):
    return np.array(draw(st.lists(st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False), min_size=dim, max_size=dim)), dtype=float)


@given(dim=st.integers(min_value=3, max_value=10))
@settings(deadline=None)
def test_metric_distance_basic_invariants(dim: int):
    np.random.seed(123)
    met = SymbolicManifoldMetric(D=dim, rank=min(4, dim))
    x = np.random.uniform(-10, 10, size=(dim,))
    mu = np.random.uniform(-10, 10, size=(dim,))
    v = np.random.uniform(-5, 5, size=(dim,))

    d1, _ = met.distance(x, mu)
    d2, _ = met.distance(mu, x)
    assert d1 >= 0.0
    assert math.isclose(d1, d2, rel_tol=1e-12, abs_tol=1e-12)

    # Re-centering invariance: depends only on x - mu
    d3, _ = met.distance(x + v, mu + v)
    assert math.isclose(d1, d3, rel_tol=1e-9, abs_tol=1e-9)

