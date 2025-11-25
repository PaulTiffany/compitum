from __future__ import annotations

import numpy as np
from hypothesis import given, strategies as st


def _topk_regret(y: np.ndarray, scores: np.ndarray, ks: list[int]) -> list[float]:
    order_o = np.argsort(y)[::-1]
    csum = np.cumsum(y[order_o])
    order_m = np.argsort(scores)[::-1]
    out = []
    n = len(y)
    for k in ks:
        k = max(1, min(n, int(k)))
        oracle = float(csum[k - 1])
        model = float(y[order_m[:k]].sum())
        reg = max(0.0, oracle - model)
        out.append(0.0 if oracle == 0.0 else reg / abs(oracle))
    return out


@given(
    y=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=5, max_size=30),
    a=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_regret_invariant_under_positive_affine(y: list[float], a: float, b: float) -> None:
    arr = np.asarray(y, dtype=float)
    # avoid degenerate all-equal
    if np.allclose(arr, arr[0]):
        return
    # Define some base scores that correlate weakly with y
    rng = np.random.default_rng(0)
    base = arr + rng.normal(0, 0.1, size=arr.shape)
    ks = [1, min(3, len(arr)), min(5, len(arr))]
    r1 = _topk_regret(arr, base, ks)
    r2 = _topk_regret(arr, a * base + b, ks)
    assert np.allclose(r1, r2, atol=1e-12, rtol=0)

