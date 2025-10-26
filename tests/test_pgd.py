import numpy as np

from compitum.pgd import RegexPromptExtractor


def test_pgd_feature_extraction() -> None:
    extractor = RegexPromptExtractor()
    prompt = (
        "This is a test prompt with some, punctuation; and numbers 123. "
        "It also has a ```code block```."
    )
    features = extractor.extract_features(prompt)
    assert isinstance(features, np.ndarray)
    assert features.shape == (39,)
    assert np.all(np.isfinite(features))


def test_pgd_byte_string_input() -> None:
    extractor = RegexPromptExtractor()
    prompt = b"This is a byte string prompt."
    features = extractor.extract_features(prompt)
    assert isinstance(features, np.ndarray)
    assert features.shape == (39,)


def test_pgd_missing_key_fallback() -> None:
    """
    Tests that if a feature key is missing, it's safely added back as 0.
    """
    extractor = RegexPromptExtractor()
    original_keys = list(extractor._r_keys)
    try:
        # Add a temporary, non-existent key to the list of Riemannian keys
        extractor._r_keys.append("imaginary_feature_key")

        # Run feature extraction. The internal `feats` dict won't have this key.
        # The safety loop should then catch it and add it with a value of 0.
        features = extractor.extract_features("test prompt")

        # The final array should be one larger than the standard size
        # 35 standard Riemannian + 1 new key + 4 Banach = 40
        assert features.shape == (40,)

        # The value for our new key should be 0. It's the last of the Riemannian features.
        assert features[35] == 0.0

    finally:
        # Clean up to ensure no side effects on other tests
        extractor._r_keys = original_keys
