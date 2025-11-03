import numpy as np

from compitum.pgd import RegexPromptExtractor


def index_of(rx: RegexPromptExtractor, key: str) -> int:
    return list(rx._r_keys).index(key)


def test_math_signals_and_keywords_ci() -> None:
    rx = RegexPromptExtractor()
    prompt = "Proof: x^2 + y_1 = 3.14. Also, $a/b$."
    out = rx.extract_features(prompt)
    i_numbers = index_of(rx, "math_3")
    i_pow = index_of(rx, "math_4")
    i_proof = index_of(rx, "math_7")
    i_latex = index_of(rx, "math_1")
    assert out[i_numbers] >= 1.0
    assert out[i_pow] >= 1.0
    assert out[i_proof] == 1.0
    assert out[i_latex] >= 1.0


def test_semantic_proxies_lengths_ci() -> None:
    rx = RegexPromptExtractor()
    # Two long tokens (>6 chars): epsilon (7), generator (9)
    prompt = "alpha beta gamma delta epsilon generator"
    out = rx.extract_features(prompt)
    i_unique = index_of(rx, "sem_3")
    i_len = index_of(rx, "sem_4")
    i_long = index_of(rx, "sem_5")
    assert out[i_unique] == 6.0
    assert out[i_len] == 6.0
    assert out[i_long] >= 2.0

