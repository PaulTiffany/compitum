import os
import io
from contextlib import redirect_stdout

import numpy as np

from compitum.coherence import CoherenceFunctional
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.predictors import CalibratedPredictor
from compitum.capabilities import Capabilities


def _toy_predictors():
    p = CalibratedPredictor()
    # Fit trivial 1D model deterministically
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 0.5, 1.0])
    p.fit(X, y)
    return {"quality": p, "latency": p, "cost": p}


def test_energy_compute_debug_prints_when_env_set():
    os.environ["COMPITUM_DEBUG_ENERGY"] = "1"
    try:
        energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.1, beta_d=0.05, beta_s=0.01)
        metric = SymbolicManifoldMetric(D=1, rank=1, delta=1e-3)
        coherence = CoherenceFunctional(k=10)
        model = Model(name="fast", center=np.array([0.0]), capabilities=Capabilities({"US"}, {"none"}), cost=0.01)
        predictors = _toy_predictors()

        x = np.array([0.2])
        buf = io.StringIO()
        with redirect_stdout(buf):
            # _step starts at 0 so modulus condition is true
            U, U_sigma, comps = energy.compute(x, model, predictors, coherence, metric)
        out = buf.getvalue()
        assert "DEBUG: Model:" in out  # printed diagnostic present
        assert isinstance(U, float) and isinstance(U_sigma, float)
        assert set(comps.keys()) >= {"quality", "latency", "cost", "distance", "evidence", "uncertainty"}
    finally:
        os.environ.pop("COMPITUM_DEBUG_ENERGY", None)


def test_energy_batch_compute_prints_on_step_multiple():
    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.1, beta_d=0.05, beta_s=0.01)
    metric = SymbolicManifoldMetric(D=1, rank=1, delta=1e-3)
    coherence = CoherenceFunctional(k=10)
    model = Model(name="fast", center=np.array([0.0]), capabilities=Capabilities({"US"}, {"none"}), cost=0.01)
    predictors = _toy_predictors()

    # Arrange for _step % 100 == 0 by setting _step=99 then processing 1 sample
    energy._step = 99  # type: ignore[attr-defined]
    x_batch = np.array([[0.1]])
    buf = io.StringIO()
    with redirect_stdout(buf):
        U_batch, U_sigma_batch, comps_list = energy.batch_compute(x_batch, model, predictors, coherence, metric)
    # Expect a timing print due to step multiple of 100
    assert "SymbolicFreeEnergy.batch_compute took" in buf.getvalue()
    assert U_batch.shape == (1,) and U_sigma_batch.shape == (1,) and len(comps_list) == 1


def test_energy_compute_timing_print_on_step_multiple():
    os.environ["COMPITUM_DEBUG_ENERGY"] = "1"
    try:
        energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.1, beta_d=0.05, beta_s=0.01)
        metric = SymbolicManifoldMetric(D=1, rank=1, delta=1e-3)
        coherence = CoherenceFunctional(k=10)
        model = Model(name="fast", center=np.array([0.0]), capabilities=Capabilities({"US"}, {"none"}), cost=0.01)
        predictors = _toy_predictors()

        # Arrange _step so that after increment it hits a multiple of 100
        energy._step = 99  # type: ignore[attr-defined]
        x = np.array([0.2])
        buf = io.StringIO()
        with redirect_stdout(buf):
            energy.compute(x, model, predictors, coherence, metric)
        assert "SymbolicFreeEnergy.compute took" in buf.getvalue()
    finally:
        os.environ.pop("COMPITUM_DEBUG_ENERGY", None)
