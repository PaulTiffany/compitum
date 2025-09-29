from unittest.mock import MagicMock, patch

import numpy as np

from compitum.coherence import CoherenceFunctional, WeightedReservoir


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
    with patch.object(coherence, '_fit', wraps=coherence._fit) as mock_fit:
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
