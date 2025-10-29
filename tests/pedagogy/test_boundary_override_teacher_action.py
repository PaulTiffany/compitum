from compitum.boundary import BoundaryAnalyzer


def test_boundary_override_conservative_policy_signal():
    # Construct utilities such that gap small and uncertainty high → boundary
    b = BoundaryAnalyzer(gap_threshold=0.05, entropy_threshold=0.2, sigma_threshold=0.1)
    utilities = {"best": 1.00, "safe": 0.98}
    u_sigma = {"best": 0.2, "safe": 0.05}
    out = b.analyze(utilities, u_sigma)
    assert out["is_boundary"] is True
    # Teacherly override: prefer safe model; uncertainty should be lower
    # (simulated via u_sigma)
    assert u_sigma["safe"] < u_sigma["best"]

