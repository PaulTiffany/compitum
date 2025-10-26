import numpy as np

from compitum.embedding_pgd import EmbeddingPGDExtractor


def test_embedding_pgd_extractor_init():
    """
    Tests that the EmbeddingPGDExtractor can be initialized.
    """
    extractor = EmbeddingPGDExtractor()
    assert extractor is not None


def test_extract_features():
    """
    Tests that the extract_features method returns the embedding in a dictionary.
    """
    extractor = EmbeddingPGDExtractor()
    embedding = np.array([1.0, 2.0, 3.0])
    features = extractor.extract_features(embedding)
    assert "embedding" in features
    np.testing.assert_array_equal(features["embedding"], embedding)


def test_extract_features_with_different_embedding():
    """
    Tests the extract_features method with a different embedding.
    """
    extractor = EmbeddingPGDExtractor()
    embedding = np.array([4.0, 5.0, 6.0, 7.0])
    features = extractor.extract_features(embedding)
    assert "embedding" in features
    np.testing.assert_array_equal(features["embedding"], embedding)


def test_extract_features_return_type():
    """
    Tests that the extract_features method returns a dictionary.
    """
    extractor = EmbeddingPGDExtractor()
    embedding = np.array([1.0, 2.0, 3.0])
    features = extractor.extract_features(embedding)
    assert isinstance(features, dict)
