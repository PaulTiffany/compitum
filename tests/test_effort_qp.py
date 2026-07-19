from compitum.effort_qp import solve_effort_1d


def test_solve_effort_1d_positive_grad() -> None:
    """Test case where the gradient is positive, expecting effort to be 1.0."""
    # grad = alpha*q1 - bt*t1 - bc*c1 = 0.5*1 - 0.1*1 - 0.1*1 = 0.3 > 0
    beta = (0.5, 0.1, 0.1)  # alpha, bt, bc
    e_star, lambdas = solve_effort_1d(q0=0, q1=1, t0=0, t1=1, c0=0, c1=1, beta=beta)
    assert e_star == 1.0
    assert lambdas["lambda_low"] == 0.0
    assert lambdas["lambda_high"] > 0.0


def test_solve_effort_1d_negative_grad() -> None:
    """Test case where the gradient is negative, expecting effort to be 0.0."""
    # grad = alpha*q1 - bt*t1 - bc*c1 = 0.1*1 - 0.5*1 - 0.5*1 = -0.9 < 0
    beta = (0.1, 0.5, 0.5)  # alpha, bt, bc
    e_star, lambdas = solve_effort_1d(q0=0, q1=1, t0=0, t1=1, c0=0, c1=1, beta=beta)
    assert e_star == 0.0
    assert lambdas["lambda_low"] > 0.0
    assert lambdas["lambda_high"] == 0.0


def test_solve_effort_1d_zero_grad_boundary() -> None:
    """grad == 0 exactly: the > vs >= choice at the boundary must resolve to
    e_star = 0.0 (indifferent -> spend no effort), not 1.0."""
    # grad = alpha*q1 - bt*t1 - bc*c1 = 0.5*1 - 0.5*1 - 0.0*1 = 0.0
    beta = (0.5, 0.5, 0.0)
    e_star, lambdas = solve_effort_1d(q0=0, q1=1, t0=0, t1=1, c0=0, c1=1, beta=beta)
    assert e_star == 0.0
    assert lambdas["lambda_low"] == 0.0
    assert lambdas["lambda_high"] == 0.0
