from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from sklearn.covariance import LedoitWolf

from .control import SRMFController


class SymbolicManifoldMetric:
    def __init__(self, D: int, rank: int, delta: float = 1e-3) -> None:
        # BRIDGEBLOCK_START alg:init_low_rank_factor
        self.D, self.rank, self.delta = D, rank, delta
        self.L = np.random.randn(D, rank) * 0.01
        self.W: Optional[np.ndarray] = None
        self.shrink = LedoitWolf()
        self.whitened_residuals: list[np.ndarray] = []
        # BRIDGEBLOCK_END alg:init_low_rank_factor

    def metric_matrix(self) -> np.ndarray:
        # BRIDGEBLOCK_START eq:spd_metric_matrix
        from .symbolic import SymbolicMatrix, SymbolicScalar

        L = SymbolicMatrix(name="L", value=self.L)
        delta = SymbolicScalar(name=r"\delta", value=self.delta)
        identity_matrix = SymbolicMatrix(name="I", value=np.eye(self.D))

        M_expression = (L @ L.T) + (delta * identity_matrix)

        # The M_expression object now contains both the computation and its own
        # LaTeX representation, which can be accessed via M_expression.to_latex()
        return M_expression.evaluate()
        # BRIDGEBLOCK_END eq:spd_metric_matrix

    # BRIDGEBLOCK_START alg:cholesky_decomposition
    def _update_cholesky(self) -> np.ndarray:
        try:
            self.W = linalg.cholesky(self.metric_matrix(), lower=False)
        except (linalg.LinAlgError, np.linalg.LinAlgError):
            print(f"Caught LinAlgError. Old delta: {self.delta}")
            self.delta = min(max(self.delta + 1e-3, 1e-5), 1e-1)
            print(f"New delta: {self.delta}")
            print(f"New metric_matrix: {self.metric_matrix()}")
            self.W = linalg.cholesky(self.metric_matrix(), lower=False)
        return self.W
    # BRIDGEBLOCK_END alg:cholesky_decomposition

    # BRIDGEBLOCK_START eq:whitened_distance
    def distance(self, x: np.ndarray, mu: np.ndarray) -> Tuple[float, float]:
        if self.W is None:
            self._update_cholesky()
        z = x - mu
        wz = self.W @ z
        d = float(np.linalg.norm(wz))
        if len(self.whitened_residuals) > self.rank:
            cov = self.shrink.fit(np.array(self.whitened_residuals)).covariance_
            sigma = float(np.sqrt(max(wz.T @ cov @ wz, 0.0)))
        else:
            sigma = 0.1
        return d, sigma
    # BRIDGEBLOCK_END eq:whitened_distance

    # BRIDGEBLOCK_START alg:metric_update_step
    def update_spd(self, x: np.ndarray, mu: np.ndarray, beta_d: float, d: float, eta: float,
                   srmf_controller: SRMFController) -> float:
        z = x - mu
        A = -(beta_d / (2 * max(d, 1e-8))) * np.outer(z, z)  # dU/dM
        grad_L = 2 * A @ self.L
        grad_norm = float(np.linalg.norm(grad_L, 2))
        eta_cap, _ = srmf_controller.update(d_star=d, grad_norm=grad_norm)
        self.L -= min(eta, eta_cap) * grad_L
        fnorm = np.linalg.norm(self.L, "fro")
        if fnorm > 10.0:
            self.L *= (10.0 / fnorm)
        W = self._update_cholesky()
        self.whitened_residuals.append(W @ z)
        if len(self.whitened_residuals) > 100:
            self.whitened_residuals.pop(0)
        return grad_norm
    # BRIDGEBLOCK_END alg:metric_update_step
