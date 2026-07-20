from __future__ import annotations

import numpy as np
import pytest

try:
    from hypothesis import given, example
    from hypothesis import strategies as st
except Exception:
    pytest.skip("hypothesis not installed", allow_module_level=True)

from compitum.pgd import RegexPromptExtractor
from compitum.utils import split_features


def _extract(ex: RegexPromptExtractor, s: str) -> np.ndarray:
    return ex.extract_features(s)


@pytest.mark.invariants
@given(text=st.text(min_size=0, max_size=200))
@example(text="")
@example(text="Hello world.")
def test_extractor_dimensions_and_banach_defaults(text: str) -> None:
    ex = RegexPromptExtractor()
    v = _extract(ex, text)
    assert v.dtype == np.float32
    assert v.shape == (35 + 4,)
    xR, xB = split_features(v)
    assert xR.shape == (35,)
    assert xB.shape == (4,)
    # Banach defaults
    assert np.allclose(xB, np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))


@pytest.mark.invariants
@given(text=st.text(min_size=0, max_size=200))
def test_bytes_equivalence(text: str) -> None:
    ex = RegexPromptExtractor()
    v1 = _extract(ex, text)
    v2 = ex.extract_features(text.encode("utf-8"))
    assert np.allclose(v1, v2)


@pytest.mark.invariants
@given(n=st.integers(min_value=1, max_value=5))
def test_sentence_count_monotonic(n: int) -> None:
    ex = RegexPromptExtractor()
    base = "Hello."
    v0 = _extract(ex, base)
    v1 = _extract(ex, base + (" Next." * n))
    # syn_2 = number of sentences; index 2 given construction order
    assert v1[2] >= v0[2] + n - 0.1  # allow float rounding


@pytest.mark.invariants
def test_code_block_increases_code_features() -> None:
    ex = RegexPromptExtractor()
    base = "Explain quicksort."
    v0 = _extract(ex, base)
    code = """
```python
for i in range(3):
    pass
```
"""
    v1 = _extract(ex, base + code)
    # code_0 (blocks) and code_1 (language hits) should not decrease
    # code_0 is at index 6+8=14? Actual order: syn(6), math(8)=14, code start at 14
    code0_idx = 14
    code1_idx = 15
    assert v1[code0_idx] >= v0[code0_idx]
    assert v1[code1_idx] >= v0[code1_idx]


@pytest.mark.invariants
def test_math_markers_increase() -> None:
    ex = RegexPromptExtractor()
    base = "State and prove a lemma."
    v0 = _extract(ex, base)
    math_snip = " Prove: $x+y$ theorem 123 ^ _ \\frac{}{}"
    v1 = _extract(ex, base + math_snip)
    # Check several math_* positions relative to start index 6
    m_start = 6
    # At least one math feature increases
    increases = 0
    for idx in (m_start + i for i in [0, 1, 2, 3, 4, 5, 6, 7]):
        if v1[idx] > v0[idx]:
            increases += 1
    assert increases >= 1


@pytest.mark.invariants
def test_length_cap() -> None:
    ex = RegexPromptExtractor()
    too_long = "a" * 10000
    v = _extract(ex, too_long)
    # syn_5 is length proxy capped at 4096 (index 5)
    assert v[5] == 4096.0


@pytest.mark.invariants
@given(k=st.integers(min_value=1, max_value=5))
def test_unique_tokens_non_decreasing(k: int) -> None:
    ex = RegexPromptExtractor()
    base = "word word word"
    v0 = _extract(ex, base)
    added = " ".join(f"UNIQ{i}" for i in range(k))
    v1 = _extract(ex, base + " " + added)
    # sem_3 is unique token count; index = syn(6) + math(8) + code(7) + sem index 0 => 6+8+7=21
    sem3_idx = 21 + 3
    sem4_idx = 21 + 4  # token count
    assert v1[sem3_idx] >= v0[sem3_idx]
    assert v1[sem4_idx] >= v0[sem4_idx]
