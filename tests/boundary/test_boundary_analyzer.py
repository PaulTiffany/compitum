from compitum.boundary import BoundaryAnalyzer


def test_boundary_flags_when_gap_small_and_sigma_high():
    b = BoundaryAnalyzer(gap_threshold=0.05, entropy_threshold=10.0, sigma_threshold=0.1)
    utilities = {"a": 1.00, "b": 0.98, "c": 0.5}
    u_sigma = {"a": 0.2, "b": 0.2, "c": 0.1}
    out = b.analyze(utilities, u_sigma)
    assert out["is_boundary"] is True


def test_boundary_clears_when_gap_large_and_sigma_low():
    b = BoundaryAnalyzer(gap_threshold=0.01, entropy_threshold=10.0, sigma_threshold=0.5)
    utilities = {"a": 1.00, "b": 0.80}
    u_sigma = {"a": 0.1, "b": 0.1}
    out = b.analyze(utilities, u_sigma)
    assert out["is_boundary"] is False

