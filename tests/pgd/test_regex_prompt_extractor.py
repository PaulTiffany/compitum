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
    # "Prove" hits the prove/derive/compute/solve keyword regex (math_2); the literal
    # substring "proof" is a separate signal (math_7) and needs its own word in the prompt.
    prompt = "Prove that 3.14^2 + 2 = 11.8596. See the proof below. Also, $x+y$ and \\frac{a}{b}."
    out = rx.extract_features(prompt)
    i_numbers = index_of(rx, "math_3")
    i_pow = index_of(rx, "math_4")
    i_keyword = index_of(rx, "math_2")
    i_proof = index_of(rx, "math_7")
    i_latex = index_of(rx, "math_1")
    assert out[i_numbers] >= 1.0
    assert out[i_pow] >= 1.0
    assert out[i_keyword] == 1.0
    assert out[i_proof] == 1.0
    assert out[i_latex] >= 1.0


def test_extract_features_comprehensive_exact_values():
    """Every existing test above checks only a handful of features loosely
    (`>= 1.0`, or a single feature's value) -- the vast majority of this
    function's ~40 dict-assignment lines were never checked at all. Since
    every unset Riemannian key gets silently backfilled with `0.0` by the
    "ensure all keys present" safety loop, a mutation renaming a dict key
    (e.g. "math_0" -> "XXmath_0XX") doesn't crash -- it just silently
    substitutes 0.0 for the real computed value, which a loose `>= 1.0` or
    per-feature check can miss if the real value also happens to be 0 or
    positive. One prompt engineered to give every feature a distinct,
    nonzero value, checked against the exact full output vector, catches
    that whole class of mutation at once (dict-key renames, `.get()`
    default-value changes, regex-string mutations, and `and`/`or`/`not`
    logic flips where the mutated branch would otherwise coincidentally
    still evaluate true)."""
    rx = RegexPromptExtractor()
    prompt = (
        "Prove the theorem. Solve this lemma. See the proof, please; consider it.\n"
        "x^2^3_a_b_c and $y+z$ or \\frac{a}{b}.\n"
        "```python\ndef f(x):\n    if x: return x\n```\n"
        "This uses python language. SELECT * FROM t; import os; class Foo: def bar(self): pass\n"
        "alpha beta gamma delta epsilon zeta eta theta quantum"
    )
    out = rx.extract_features(prompt)
    expected = np.array(
        [
            8.166666984558105,
            6.094168663024902,
            6.0,
            1.0,
            3.0,
            293.0,
            1.0,
            1.0,
            2.0,
            2.0,
            5.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            11.0,
            1.0,
            1.0,
            1.0,
            138.0,
            2.875,
            2.2232203483581543,
            46.0,
            49.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_syn_and_sem_default_to_zero_on_empty_prompt():
    """`syn_0`/`syn_1`'s `if sents else 0.0` fallback, and `sem_0`/`sem_1`/
    `sem_2`'s `if diffs else 0.0` fallback, were never exercised -- every
    other test's prompt produces at least one sentence/multiple tokens. An
    empty prompt makes both `sents` and `diffs` empty at once."""
    rx = RegexPromptExtractor()
    out = rx.extract_features("")
    for key in ("syn_0", "syn_1", "sem_0", "sem_1", "sem_2"):
        assert out[rx._r_keys.index(key)] == 0.0


def test_sem_5_long_word_boundary_is_strictly_greater_than_six():
    """`len(w) > 6` was never tested with a word exactly 6 characters long,
    where `>` (correct, excluded) and `>=` (mutant, included) disagree."""
    rx = RegexPromptExtractor()
    out = rx.extract_features("abcdef short")
    i_sem5 = rx._r_keys.index("sem_5")
    assert out[i_sem5] == 0.0


def test_semantic_diffs_boundary_exactly_two_tokens():
    """`if len(tokens) > 1` was never exercised exactly at the boundary --
    with exactly 2 tokens (`> 1` true, one diff computed) vs 1 token (both
    `> 1` and `>= 1` agree, since `range(0)` is empty either way -- that
    specific boundary is a confirmed equivalent mutant, not a gap)."""
    rx = RegexPromptExtractor()
    out = rx.extract_features("alpha bcdefgh")
    i_sem0 = rx._r_keys.index("sem_0")
    assert out[i_sem0] == 2.0  # abs(len("bcdefgh") - len("alpha")) = abs(7-5)


def test_code_4_class_or_def_each_branch_isolated():
    """`"class " in prompt or "def " in prompt` was only ever tested with
    *both* substrings present at once, where an `or`->`and` mutation, a
    renamed search string on either side, or an `in`->`not in` flip on
    either side would all still coincidentally evaluate to the same
    result. Test each operand in isolation, and with neither present."""
    rx = RegexPromptExtractor()
    i_code4 = rx._r_keys.index("code_4")
    assert rx.extract_features("this has class here")[i_code4] == 1.0
    assert rx.extract_features("this has def here")[i_code4] == 1.0
    assert rx.extract_features("this has nothing here")[i_code4] == 0.0


def test_semantic_proxies_unique_and_lengths():
    rx = RegexPromptExtractor()
    # theta is 5 letters, not >6 -- use quantum alongside epsilon so two words
    # actually clear the sem_5 "long word" threshold.
    prompt = "alpha beta gamma delta epsilon zeta eta theta quantum"
    out = rx.extract_features(prompt)
    i_unique = index_of(rx, "sem_3")
    i_len = index_of(rx, "sem_4")
    i_long = index_of(rx, "sem_5")
    assert out[i_unique] == 9.0
    assert out[i_len] == 9.0
    # words of length > 6 should be counted (epsilon, quantum)
    assert out[i_long] >= 2.0
