from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from sklearn.covariance import LedoitWolf

from .control import LyapunovController


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
        except linalg.LinAlgError:
            print(f"Caught LinAlgError. Old delta: {self.delta}")
            self.delta = min(max(self.delta + 1e-3, 1e-5), 1e-1)
            print(f"New delta: {self.delta}")
            print(f"New metric_matrix: {self.metric_matrix()}")
            self.W = linalg.cholesky(self.metric_matrix(), lower=False)
        except np.linalg.LinAlgError:  # pragma: no cover - symmetric to LinAlgError above
            print(f"Caught LinAlgError. Old delta: {self.delta}")
            self.delta = min(max(self.delta + 1e-3, 1e-5), 1e-1)
            print(f"New delta: {self.delta}")
            print(f"New metric_matrix: {self.metric_matrix()}")
            self.W = linalg.cholesky(self.metric_matrix(), lower=False)
        return self.W

    # BRIDGEBLOCK_END alg:cholesky_decomposition

    # BRIDGEBLOCK_START eq:whitened_distance
    def distance(self, x: np.ndarray, mu: np.ndarray) -> Tuple[float, float]:
        if os.environ.get("COMPITUM_DEBUG_METRIC") == "1":
            print("!!! DISTANCE METHOD CALLED !!!")
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

    def batch_distance(self, x_batch: np.ndarray, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.W is None:
            self._update_cholesky()
        if self.W is None:  # Check again after update
            raise ValueError("self.W should not be None after _update_cholesky()")
        W_matrix = self.W  # Explicitly assign to a local variable to help mypy infer type

        z_batch = x_batch - mu  # Broadcasting handles this
        wz_batch = z_batch @ W_matrix.T  # W is upper triangular, so W @ z is z @ W.T

        d_batch = np.linalg.norm(wz_batch, axis=1)  # Norm along each row

        # Sigma calculation is complex due to LedoitWolf. For batching, simplify for now.
        # This part might need further optimization or a different approach for true batching.
        if len(self.whitened_residuals) > self.rank:
            # Compute covariance once from historical residuals
            cov = self.shrink.fit(np.array(self.whitened_residuals)).covariance_

            # Vectorize quadratic form: wz.T @ cov @ wz for each wz in wz_batch
            # np.einsum('ij,jk,ik->i', wz_batch, cov, wz_batch) computes
            # sum_j sum_k wz_ij * cov_jk * wz_ik
            # This is equivalent to np.diag(wz_batch @ cov @ wz_batch.T)
            sigma_squared_batch = np.einsum("ij,jk,ik->i", wz_batch, cov, wz_batch)
            sigma_batch = np.sqrt(np.maximum(sigma_squared_batch, 0.0))
        else:
            sigma_batch = np.full_like(d_batch, 0.1)  # Default sigma for all samples

        return d_batch, sigma_batch

    # BRIDGEBLOCK_START alg:metric_update_step
    def update_spd(
        self,
        x: np.ndarray,
        mu: np.ndarray,
        beta_d: float,
        d: float,
        eta: float,
        srmf_controller: LyapunovController,
    ) -> float:
        return self.batch_update_spd(
            x_batch=np.array([x]),
            mu_batch=np.array([mu]),
            beta_d=beta_d,
            d_batch=np.array([d]),
            eta=eta,
            srmf_controller=srmf_controller,
        )

    # BRIDGEBLOCK_END alg:metric_update_step

    def batch_update_spd(
        self,
        x_batch: np.ndarray,
        mu_batch: np.ndarray,
        beta_d: float,
        d_batch: np.ndarray,
        eta: float,
        srmf_controller: LyapunovController,
    ) -> float:
        batch_size = x_batch.shape[0]
        if batch_size == 0:
            return 1.0

        z_batch = x_batch - mu_batch

        d_batch_safe = np.maximum(d_batch, 1e-8)
        # Vectorized computation of A_batch and grad_L_batch
        A_batch = (
            beta_d
            / (2 * d_batch_safe[:, np.newaxis, np.newaxis])
        ) * np.einsum("bi,bj->bij", z_batch, z_batch)

        # Sum gradients over the batch
        grad_L = 2 * np.sum(np.einsum("bij,jk->bik", A_batch, self.L), axis=0)

        grad_norm = float(np.linalg.norm(grad_L, 2))

        # For the SRMF controller, we can either pass the average d_star or the last one.
        # Let's use the average for a more stable update.
        avg_d_star = np.mean(d_batch)
        eta_cap, _ = srmf_controller.update(d_star=avg_d_star, grad_norm=grad_norm)

        # Stability cap using average z_norm2
        z_norm2_batch = np.sum(z_batch * z_batch, axis=1)
        avg_z_norm2 = np.mean(z_norm2_batch)
        lipschitz = max(beta_d * avg_z_norm2, 1e-8)
        eta_stab = 1.0 / lipschitz
        eta_eff = min(eta, eta_cap, eta_stab)

        def surrogate_energy(L: np.ndarray) -> float:
            v_batch = np.einsum("jd,bd->bj", L.T, z_batch)
            return 0.5 * beta_d * np.sum(np.sum(v_batch * v_batch, axis=1))

        e0 = surrogate_energy(self.L)
        new_L = self.L - eta_eff * grad_L
        e1 = surrogate_energy(new_L)

        if e1 > e0:
            bt = 0
            while bt < 8 and e1 > e0:
                eta_eff *= 0.5
                new_L = self.L - eta_eff * grad_L
                e1 = surrogate_energy(new_L)
                bt += 1
            if e1 > e0:  # pragma: no cover - defensive fallback
                eta_eff = 0.0  # pragma: no cover
                new_L = self.L  # pragma: no cover

        self.L = new_L
        fnorm = np.linalg.norm(self.L, "fro")
        if fnorm > 10.0:
            self.L *= 10.0 / fnorm

        W = self._update_cholesky()

        # Update whitened_residuals with the batch
        wz_batch = np.einsum('ij,bj->bi', W, z_batch)
        self.whitened_residuals.extend(wz_batch)
        while len(self.whitened_residuals) > 100:
            self.whitened_residuals.pop(0)

        return grad_norm
