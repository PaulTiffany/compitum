from typing import Dict, Tuple


def solve_effort_1d(
    q0: float, q1: float, t0: float, t1: float, c0: float, c1: float,
    beta: Tuple[float, float, float]
) -> Tuple[float, Dict[str, float]]:
    """
    Linearized effort e∈[0,1] around e0. Returns e_star and box multipliers.
    U(e) = α(q0+q1 e) - βt(t0+t1 e) - βc(c0+c1 e) + const
    """
    alpha, bt, bc = beta
    grad = alpha*q1 - bt*t1 - bc*c1
    e_star = 1.0 if grad > 0 else 0.0
    lam_low  = max(0.0, -grad) if e_star == 0.0 else 0.0
    lam_high = max(0.0,  grad) if e_star == 1.0 else 0.0
    return float(e_star), {"lambda_low": lam_low, "lambda_high": lam_high}
