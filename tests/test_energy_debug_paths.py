import os
import re
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
        model = Model(
            name="fast",
            center=np.array([0.0]),
            capabilities=Capabilities({"US"}, {"none"}),
            cost=0.01,
        )
        predictors = _toy_predictors()

        x = np.array([0.2])
        buf = io.StringIO()
        with redirect_stdout(buf):
            # _step starts at 0 so modulus condition is true
            U, U_sigma, comps = energy.compute(x, model, predictors, coherence, metric)
        out = buf.getvalue()
        assert "DEBUG: Model:" in out  # printed diagnostic present
        assert isinstance(U, float) and isinstance(U_sigma, float)
        assert set(comps.keys()) >= {
            "quality",
            "latency",
            "cost",
            "distance",
            "evidence",
            "uncertainty",
        }

        # Only checking a substring above leaves every other static label and
        # variable in both debug print statements unverified -- a mutation to
        # any of them (wrong label text, swapped variable, dropped field)
        # would survive. Reconstruct the two lines' exact expected content
        # from the actual returned comps/energy state (not hardcoded values,
        # since metric.distance()'s internal randomness makes d/log_e
        # non-reproducible across runs) and assert full equality.
        q0 = comps["quality"]
        t0 = -comps["latency"]
        c0 = -comps["cost"] - model.cost
        d = -comps["distance"]
        log_e = comps["evidence"]
        expected_line1 = (
            f"DEBUG: Model: {model.name}, q0: {q0}, t0: {t0}, c0: {c0}, "
            f"cost: {model.cost}, d: {d}, log_e: {log_e}"
        )
        expected_line2 = (
            f"DEBUG: alpha: {energy.alpha}, beta_t: {energy.beta_t}, beta_c: {energy.beta_c}, "
            f"beta_d: {energy.beta_d}, beta_s: {energy.beta_s}"
        )
        assert out == expected_line1 + "\n" + expected_line2 + "\n"
    finally:
        os.environ.pop("COMPITUM_DEBUG_ENERGY", None)


def test_energy_batch_compute_prints_on_step_multiple():
    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.1, beta_d=0.05, beta_s=0.01)
    metric = SymbolicManifoldMetric(D=1, rank=1, delta=1e-3)
    coherence = CoherenceFunctional(k=10)
    model = Model(
        name="fast", center=np.array([0.0]), capabilities=Capabilities({"US"}, {"none"}), cost=0.01
    )
    predictors = _toy_predictors()

    # Arrange for _step % 100 == 0 by setting _step=99 then processing 1 sample
    energy._step = 99  # type: ignore[attr-defined]
    x_batch = np.array([[0.1]])
    buf = io.StringIO()
    with redirect_stdout(buf):
        U_batch, U_sigma_batch, comps_list = energy.batch_compute(
            x_batch, model, predictors, coherence, metric
        )
    # Expect a timing print due to step multiple of 100. The elapsed-time
    # value is inherently non-deterministic, so match the full line
    # structure via regex rather than just a leading substring.
    out = buf.getvalue()
    assert re.fullmatch(
        r"SymbolicFreeEnergy\.batch_compute took \d+\.\d{4} seconds for 1 samples\n", out
    )
    # The regex alone would still match `time.time() + start_time` (a huge
    # unix-epoch-scale number formatted with 4 decimals looks the same
    # syntactically) -- bound the elapsed value to catch that sign flip.
    elapsed = float(re.search(r"took (\d+\.\d{4}) seconds", out).group(1))  # type: ignore[union-attr]
    assert elapsed < 5.0
    assert U_batch.shape == (1,) and U_sigma_batch.shape == (1,) and len(comps_list) == 1


def test_energy_compute_timing_print_on_step_multiple():
    os.environ["COMPITUM_DEBUG_ENERGY"] = "1"
    try:
        energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.1, beta_d=0.05, beta_s=0.01)
        metric = SymbolicManifoldMetric(D=1, rank=1, delta=1e-3)
        coherence = CoherenceFunctional(k=10)
        model = Model(
            name="fast",
            center=np.array([0.0]),
            capabilities=Capabilities({"US"}, {"none"}),
            cost=0.01,
        )
        predictors = _toy_predictors()

        # Arrange _step so that after increment it hits a multiple of 100
        energy._step = 99  # type: ignore[attr-defined]
        x = np.array([0.2])
        buf = io.StringIO()
        with redirect_stdout(buf):
            energy.compute(x, model, predictors, coherence, metric)
        out = buf.getvalue()
        # Two DEBUG lines (env-gated) precede the timing line here since
        # COMPITUM_DEBUG_ENERGY is set; match the timing line's full
        # structure via regex (elapsed time is non-deterministic).
        last_line = out.splitlines(keepends=True)[-1]
        assert re.search(r"^SymbolicFreeEnergy\.compute took \d+\.\d{4} seconds\n\Z", last_line)
        # The regex alone would still match `time.time() + start_time` (a
        # huge epoch-scale number still has 4 decimals) -- bound it.
        elapsed = float(re.search(r"took (\d+\.\d{4}) seconds", last_line).group(1))  # type: ignore[union-attr]
        assert elapsed < 5.0
    finally:
        os.environ.pop("COMPITUM_DEBUG_ENERGY", None)


def test_energy_compute_debug_gate_checked_before_increment_at_step_100():
    """The debug-print gate (`self._step % 100 == 0 and env == "1"`) is
    checked *before* `self._step` is incremented -- the existing tests only
    exercise it at `_step == 0` (a default fresh instance) or `_step == 99`
    (where it's False either way), never at a nonzero multiple of 100 before
    increment. At `_step == 0`, `% 100` and `/ 100` and `% 101` all agree
    (all equal 0), masking mutations to the modulus/operator; at `_step ==
    100` they diverge (`100/100 == 1.0 != 0`; `100 % 101 == 100 != 0`)."""
    os.environ["COMPITUM_DEBUG_ENERGY"] = "1"
    try:
        energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.1, beta_d=0.05, beta_s=0.01)
        metric = SymbolicManifoldMetric(D=1, rank=1, delta=1e-3)
        coherence = CoherenceFunctional(k=10)
        model = Model(
            name="fast",
            center=np.array([0.0]),
            capabilities=Capabilities({"US"}, {"none"}),
            cost=0.01,
        )
        predictors = _toy_predictors()

        energy._step = 100  # type: ignore[attr-defined]
        x = np.array([0.2])
        buf = io.StringIO()
        with redirect_stdout(buf):
            energy.compute(x, model, predictors, coherence, metric)
        out = buf.getvalue()
        assert "DEBUG: Model:" in out
    finally:
        os.environ.pop("COMPITUM_DEBUG_ENERGY", None)


def test_energy_compute_debug_gate_requires_env_var_even_at_step_zero():
    """The gate is `step % 100 == 0 AND env == "1"` -- with env unset, a
    fresh instance (`_step == 0`, satisfying the step half) must print
    nothing. An `and` -> `or` mutation would print regardless of the env
    var whenever the step condition alone is true."""
    os.environ.pop("COMPITUM_DEBUG_ENERGY", None)
    energy = SymbolicFreeEnergy(alpha=1.0, beta_t=0.1, beta_c=0.1, beta_d=0.05, beta_s=0.01)
    metric = SymbolicManifoldMetric(D=1, rank=1, delta=1e-3)
    coherence = CoherenceFunctional(k=10)
    model = Model(
        name="fast", center=np.array([0.0]), capabilities=Capabilities({"US"}, {"none"}), cost=0.01
    )
    predictors = _toy_predictors()

    x = np.array([0.2])
    buf = io.StringIO()
    with redirect_stdout(buf):
        energy.compute(x, model, predictors, coherence, metric)
    assert buf.getvalue() == ""
