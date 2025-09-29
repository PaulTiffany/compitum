from compitum.pgd import ProductionPGDExtractor


def test_pgd_extractor_basic() -> None:
    extractor = ProductionPGDExtractor()
    prompt = (
        "Prove that ∫(x^2)dx = x^3/3. This is a simple calculus proof. "
        "Here is some code: ```python\nprint('hello')\n```"
    )

    features = extractor.extract_features(prompt)

    assert isinstance(features, dict)

    # Check that some feature values are being calculated
    assert features["syn_0"] > 0  # Mean sentence length
    assert features["syn_2"] > 0  # Number of sentences

    # Check math features
    assert features["math_0"] > 0  # Math ops
    assert features["math_7"] > 0  # 'proof' keyword

    # Check code features
    assert features["code_0"] > 0  # Code blocks
    assert features["code_1"] > 0  # Language hits ('python')

    # Check semantic features
    assert features["sem_3"] > 0  # Unique tokens
    assert features["sem_4"] > 0  # Total tokens

    # Check pragmatic features are present
    assert "prag_latency_class" in features
