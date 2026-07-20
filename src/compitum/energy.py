from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from .coherence import CoherenceFunctional
from .metric import SymbolicManifoldMetric
from .models import Model
from .predictors import CalibratedPredictor


class SymbolicFreeEnergy:
    def __init__(
        self, alpha: float, beta_t: float, beta_c: float, beta_d: float, beta_s: float
    ) -> None:
        self.alpha = alpha
        self.beta_t = beta_t
        self.beta_c = beta_c
        self._beta_d = beta_d
        self.beta_s = beta_s
        self._step = 0

    @property
    def beta_d(self) -> float:
        return self._beta_d

    @beta_d.setter
    def beta_d(self, v: float) -> None:
        self._beta_d = v

    def compute(
        self,
        xR: np.ndarray,
        model: Model,
        predictors: Dict[str, CalibratedPredictor],
        coherence: CoherenceFunctional,
        metric: SymbolicManifoldMetric,
    ) -> Tuple[float, float, Dict[str, float]]:
        start_time = time.time()
        d, d_std = metric.distance(xR, model.center)
        q, q_lo, q_hi = predictors["quality"].predict(np.array([xR]))
        t, t_lo, t_hi = predictors["latency"].predict(np.array([xR]))
        c, c_lo, c_hi = predictors["cost"].predict(np.array([xR]))

        # evidence in whitened space
        W = metric.W if metric.W is not None else metric._update_cholesky()
        xw = W @ (xR - model.center)
        log_e = coherence.log_evidence(model.name, xw)

        # BRIDGEBLOCK_START def:symbolic_free_energy_computation
        # Debugging: Print individual components of utility calculation
        import os as _os

        if self._step % 100 == 0 and _os.environ.get("COMPITUM_DEBUG_ENERGY") == "1":
            print(
                (
                    f"DEBUG: Model: {model.name}, q0: {q[0]}, t0: {t[0]}, c0: {c[0]}, "
                    f"cost: {model.cost}, d: {d}, log_e: {log_e}"
                ),
                flush=True,
            )
            print(
                (
                    f"DEBUG: alpha: {self.alpha}, beta_t: {self.beta_t}, beta_c: {self.beta_c}, "
                    f"beta_d: {self.beta_d}, beta_s: {self.beta_s}"
                ),
                flush=True,
            )

        U = (
            self.alpha * q[0]
            - self.beta_t * t[0]
            - self.beta_c * (c[0] + model.cost)
            - self.beta_d * d
            + self.beta_s * log_e
        )
        # BRIDGEBLOCK_END def:symbolic_free_energy_computation
        U_var = (
            (self.alpha * (q_hi[0] - q_lo[0]) / 3.92) ** 2
            + (self.beta_t * (t_hi[0] - t_lo[0]) / 3.92) ** 2
            + (self.beta_c * (c_hi[0] - c_lo[0]) / 3.92) ** 2
            + (self.beta_d * d_std) ** 2
        )
        comps = {
            "quality": float(q[0]),
            "latency": float(-t[0]),
            "cost": float(-(c[0] + model.cost)),
            "distance": float(-d),
            "evidence": float(log_e),
            "uncertainty": float(np.sqrt(U_var)),
        }
        self._step += 1
        if self._step % 100 == 0 and _os.environ.get("COMPITUM_DEBUG_ENERGY") == "1":
            print(f"SymbolicFreeEnergy.compute took {time.time() - start_time:.4f} seconds")
        return float(U), float(np.sqrt(U_var)), comps

    def batch_compute(
        self,
        xR_batch: np.ndarray,
        model: Model,
        predictors: Dict[str, CalibratedPredictor],
        coherence: CoherenceFunctional,
        metric: SymbolicManifoldMetric,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        start_time = time.time()
        num_samples = xR_batch.shape[0]

        # Batch distance computation
        d_batch, d_std_batch = metric.batch_distance(xR_batch, model.center)

        # Batch predictor calls
        q_batch, q_lo_batch, q_hi_batch = predictors["quality"].predict(xR_batch)
        t_batch, t_lo_batch, t_hi_batch = predictors["latency"].predict(xR_batch)
        c_batch, c_lo_batch, c_hi_batch = predictors["cost"].predict(xR_batch)

        # Batch evidence computation
        W = metric.W if metric.W is not None else metric._update_cholesky()
        xw_batch = (W @ (xR_batch - model.center).T).T  # Apply W to each sample in batch
        log_e_batch = coherence.batch_log_evidence(model.name, xw_batch)

        # Vectorized free energy computation
        U_batch = (
            self.alpha * q_batch
            - self.beta_t * t_batch
            - self.beta_c * (c_batch + model.cost)
            - self.beta_d * d_batch
            + self.beta_s * log_e_batch
        )

        U_var_batch = (
            (self.alpha * (q_hi_batch - q_lo_batch) / 3.92) ** 2
            + (self.beta_t * (t_hi_batch - t_lo_batch) / 3.92) ** 2
            + (self.beta_c * (c_hi_batch - c_lo_batch) / 3.92) ** 2
            + (self.beta_d * d_std_batch) ** 2
        )

        comps_list: List[Dict[str, float]] = []
        for i in range(num_samples):
            comps_list.append(
                {
                    "quality": float(q_batch[i]),
                    "latency": float(-t_batch[i]),
                    "cost": float(-(c_batch[i] + model.cost)),
                    "distance": float(-d_batch[i]),
                    "evidence": float(log_e_batch[i]),
                    "uncertainty": float(np.sqrt(U_var_batch[i])),
                }
            )

        self._step += num_samples  # Increment step by number of samples processed
        if self._step % 100 == 0:
            print(
                f"SymbolicFreeEnergy.batch_compute took {time.time() - start_time:.4f} "
                f"seconds for {num_samples} samples"
            )
        return U_batch, np.sqrt(U_var_batch), comps_list
