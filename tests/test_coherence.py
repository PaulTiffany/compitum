from unittest.mock import MagicMock, patch

import numpy as np

from compitum.coherence import CoherenceFunctional, WeightedReservoir


def test_reservoir_default_k() -> None:
    """Every other test passes `k=` explicitly -- the constructor's actual
    default (1000) was never exercised."""
    reservoir = WeightedReservoir()
    assert reservoir.k == 1000


def test_coherence_functional_default_k() -> None:
    """Same gap as the reservoir default above, one level up: every test
    constructs `CoherenceFunctional()` with no args but never checks the `k`
    it passes down to each lazily-created `WeightedReservoir`."""
    coherence = CoherenceFunctional()
    assert coherence.res["any_model"].k == 1000


def test_reservoir_add_below_k() -> None:
    reservoir = WeightedReservoir(k=5)
    for i in range(4):
        reservoir.add(np.array([i]), 1.0)
    assert len(reservoir.buf) == 4
    assert reservoir.tot_w == 4.0


def test_reservoir_add_above_k_replace() -> None:
    """Test reservoir sampling with a mocked RNG to force replacement."""
    mock_rng = MagicMock()
    # Mock the random integer generation to always return 0, forcing replacement of the first
    # element.
    mock_rng.integers.return_value = 0

    reservoir = WeightedReservoir(k=3, rng=mock_rng)
    for i in range(3):
        reservoir.add(np.array([i]), 1.0)

    assert reservoir.buf[0][0][0] == 0

    # This add should replace the element at index 0
    reservoir.add(np.array([99]), 1.0)
    mock_rng.integers.assert_called_with(0, 4)
    assert len(reservoir.buf) == 3
    assert reservoir.buf[0][0][0] == 99


def test_reservoir_add_clamps_nonpositive_weight() -> None:
    """No existing test passes a weight <= 0 -- the max(w, 1e-6) clamp that
    keeps zero/negative weights from stalling reservoir sampling was never
    exercised."""
    reservoir = WeightedReservoir(k=5)
    reservoir.add(np.array([1]), 0.0)
    assert reservoir.tot_w == 1e-6

    reservoir2 = WeightedReservoir(k=5)
    reservoir2.add(np.array([1]), -5.0)
    assert reservoir2.tot_w == 1e-6


def test_coherence_not_enough_data() -> None:
    coherence = CoherenceFunctional()
    # Add only 5 data points, less than the threshold of 10
    for i in range(5):
        coherence.update("test_model", np.array([i]), 1.0)

    evidence = coherence.log_evidence("test_model", np.array([0]))
    assert evidence == 0.0


def test_coherence_enough_data() -> None:
    coherence = CoherenceFunctional()
    rng = np.random.default_rng(0)
    # Add enough data points to trigger KDE fitting
    for _ in range(15):
        coherence.update("test_model", rng.random(2), 1.0)

    # Calling log_evidence should now fit a KDE and return a non-zero value
    evidence = coherence.log_evidence("test_model", rng.random(2))
    assert evidence != 0.0

    # Check that the KDE is now cached
    assert "test_model" in coherence.kde_cache

    # A second call should use the cache and not call _fit
    with patch.object(coherence, "_fit", wraps=coherence._fit) as mock_fit:
        coherence.log_evidence("test_model", rng.random(2))
        mock_fit.assert_not_called()


def test_reservoir_add_above_k_no_replace() -> None:
    """Test reservoir sampling where the random number is out of range, causing no replacement."""
    mock_rng = MagicMock()
    # Mock the random integer generation to return a value >= k, causing no replacement.
    mock_rng.integers.return_value = 4

    reservoir = WeightedReservoir(k=3, rng=mock_rng)
    for i in range(3):
        reservoir.add(np.array([i]), 1.0)

    # Keep a copy of the buffer before the call
    original_buf_content = [item[0][0] for item in reservoir.buf]

    # This add should NOT replace any element
    reservoir.add(np.array([99]), 1.0)
    mock_rng.integers.assert_called_with(0, 4)
    assert len(reservoir.buf) == 3
    # Assert that the buffer is unchanged
    for i in range(3):
        assert reservoir.buf[i][0][0] == original_buf_content[i]


def test_reservoir_replace_index_exactly_at_k_does_not_replace() -> None:
    """The existing mocked-RNG tests use j=0 (well inside k) and j=4 (clearly
    outside k=3) -- neither exercises j == k exactly, where `j < self.k`
    (correct, no replace) and `j <= self.k` (mutant, replaces) disagree."""
    mock_rng = MagicMock()
    mock_rng.integers.return_value = 3  # exactly k

    reservoir = WeightedReservoir(k=3, rng=mock_rng)
    for i in range(3):
        reservoir.add(np.array([i]), 1.0)
    original = [item[0][0] for item in reservoir.buf]

    reservoir.add(np.array([99]), 1.0)
    assert [item[0][0] for item in reservoir.buf] == original


def test_fit_bandwidth_matches_scott_rule_exactly() -> None:
    """No existing test checks the fitted KDE's actual bandwidth value --
    only downstream behavior (evidence != 0), which doesn't distinguish a
    changed sign or a +1 shift in Scott's rule's exponent denominator."""
    coherence = CoherenceFunctional()
    rng = np.random.default_rng(0)
    for _ in range(15):
        coherence.update("m", rng.random(2), 1.0)
    kde = coherence._fit("m")
    n, d = 15, 2
    assert kde is not None
    assert kde.bandwidth == n ** (-1.0 / (d + 4))


def test_log_evidence_clips_to_exact_bounds() -> None:
    """No existing test's KDE score naturally exceeds +-10 (Scott's-rule
    bandwidth keeps realistic log-densities well inside that range), so the
    clip's exact bounds were never exercised on either side. Mock the fitted
    KDE directly to force scores outside the clip range."""
    coherence = CoherenceFunctional()
    mock_kde = MagicMock()
    coherence.kde_cache["m"] = mock_kde

    mock_kde.score_samples.return_value = np.array([50.0])
    assert coherence.log_evidence("m", np.array([0.0])) == 10.0

    mock_kde.score_samples.return_value = np.array([-50.0])
    assert coherence.log_evidence("m", np.array([0.0])) == -10.0


def test_batch_log_evidence_clips_to_exact_bounds() -> None:
    coherence = CoherenceFunctional()
    mock_kde = MagicMock()
    coherence.kde_cache["m"] = mock_kde
    mock_kde.score_samples.return_value = np.array([50.0, -50.0])

    result = coherence.batch_log_evidence("m", np.array([[0.0], [0.0]]))
    assert list(result) == [10.0, -10.0]


def test_batch_log_evidence() -> None:
    coherence = CoherenceFunctional()
    xw_batch = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Test case with not enough data
    result_no_data = coherence.batch_log_evidence("test_model_1", xw_batch)
    assert np.all(result_no_data == 0)
    assert result_no_data.shape == (2,)

    # Add enough data for KDE to be fitted
    for i in range(15):
        coherence.update("test_model_2", np.random.rand(2), 1.0)

    # Test case with enough data
    result_with_data = coherence.batch_log_evidence("test_model_2", xw_batch)
    assert result_with_data.shape == (2,)
    assert np.any(result_with_data != 0)
