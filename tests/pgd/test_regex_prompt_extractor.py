import numpy as np

from compitum.pgd import RegexPromptExtractor


def index_of(extractor: RegexPromptExtractor, key: str) -> int:
    return extractor._r_keys.index(key)  # type: ignore[attr-defined]


def test_shape_and_banach_tail():
    rx = RegexPromptExtractor()
    out = rx.extract_features("Hello world.")
    # 35 Riemannian + 4 Banach
    assert out.shape == (39,)
    assert out.dtype == np.float32
    # Banach tail defaults
    assert np.allclose(out[-4:], np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_code_block_and_language_hits():
    rx = RegexPromptExtractor()
    prompt = """
    Here is some code:
    ```python
    def f(x):
        return x*x
    ```
    This uses Python language.
    """
    out = rx.extract_features(prompt)
    i_code_blocks = index_of(rx, "code_0")
    i_lang_hits = index_of(rx, "code_1")
    assert out[i_code_blocks] >= 1.0
    assert out[i_lang_hits] >= 1.0


def test_math_signals_and_keywords():
    rx = RegexPromptExtractor()
    prompt = "Prove that 3.14^2 + 2 = 11.8596. Also, $x+y$ and \\frac{a}{b}."
    out = rx.extract_features(prompt)
    i_numbers = index_of(rx, "math_3")
    i_pow = index_of(rx, "math_4")
    i_proof = index_of(rx, "math_7")
    i_latex = index_of(rx, "math_1")
    assert out[i_numbers] >= 1.0
    assert out[i_pow] >= 1.0
    assert out[i_proof] == 1.0
    assert out[i_latex] >= 1.0


def test_semantic_proxies_unique_and_lengths():
    rx = RegexPromptExtractor()
    prompt = "alpha beta gamma delta epsilon zeta eta theta"
    out = rx.extract_features(prompt)
    i_unique = index_of(rx, "sem_3")
    i_len = index_of(rx, "sem_4")
    i_long = index_of(rx, "sem_5")
    assert out[i_unique] == 8.0
    assert out[i_len] == 8.0
    # words of length > 6 should be counted (epsilon, theta)
    assert out[i_long] >= 2.0

