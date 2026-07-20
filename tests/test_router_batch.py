from typing import Dict, cast
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter
from compitum.utils import pgd_hash


def test_router_batch_route() -> None:
    # Setup Mocks
    model1 = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    models = [model1]
    pgd_extractor = MagicMock()
    energy = MagicMock()
    energy.batch_compute.return_value = (
        np.array([0.9, 0.95]),
        np.array([0.1, 0.1]),
        [{"distance": -0.5}, {"distance": -0.6}],
    )
    type(energy).beta_d = PropertyMock(return_value=0.5)
    metric_map = {"m1": MagicMock(spec=SymbolicManifoldMetric)}
    solver = MagicMock()
    solver.select.return_value = (model1, {"feasible": True})
    boundary = MagicMock()
    boundary.analyze.return_value = {"is_boundary": False}
    srmf = MagicMock()
    srmf.batch_update.return_value = ([], [{}, {}])
    srmf.update.return_value = (1.0, {"trust_radius": 1.0})
    coherence = MagicMock()
    predictors = {
        "m1": {
            "quality": MagicMock(spec=CalibratedPredictor),
            "latency": MagicMock(spec=CalibratedPredictor),
            "cost": MagicMock(spec=CalibratedPredictor),
        }
    }
    router = CompitumRouter(
        models=models,
        predictors=cast(Dict[str, Dict[str, CalibratedPredictor]], predictors),
        solver=solver,
        coherence=coherence,
        boundary=boundary,
        srmf=srmf,
        pgd_extractor=pgd_extractor,
        metric_map=cast(Dict[str, SymbolicManifoldMetric], metric_map),
        energy=energy,
        update_stride=1,
    )

    embeddings = np.random.rand(2, 2)
    prompts = ["prompt1", "prompt2"]
    certs = router.batch_route(embeddings, prompts)

    assert len(certs) == 2
    assert energy.batch_compute.call_count == 1
    assert solver.select.call_count == 2
    assert boundary.analyze.call_count == 2
    assert metric_map["m1"].batch_update_spd.call_count == 1
    assert srmf.batch_update.call_count == 1

    # Call it again with xB_batch to cover the other branch
    xB_batch = np.random.rand(2, 4)
    certs2 = router.batch_route(embeddings, prompts, xB_batch=xB_batch)
    assert len(certs2) == 2


def test_router_batch_route_batch_updates() -> None:
    # Setup Mocks
    model1 = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    models = [model1]
    pgd_extractor = MagicMock()
    energy = MagicMock()
    energy.batch_compute.return_value = (
        np.array([0.9, 0.95]),
        np.array([0.1, 0.1]),
        [{"distance": -0.5}, {"distance": -0.6}],
    )
    type(energy).beta_d = PropertyMock(return_value=0.5)
    metric_map = {"m1": MagicMock(spec=SymbolicManifoldMetric)}
    solver = MagicMock()
    solver.select.return_value = (model1, {"feasible": True})
    boundary = MagicMock()
    boundary.analyze.return_value = {"is_boundary": False}
    srmf = MagicMock()
    srmf.batch_update.return_value = ([], [{}, {}])  # Return two empty dicts for drift_statuses
    coherence = MagicMock()
    predictors = {
        "m1": {
            "quality": MagicMock(spec=CalibratedPredictor),
            "latency": MagicMock(spec=CalibratedPredictor),
            "cost": MagicMock(spec=CalibratedPredictor),
        }
    }
    router = CompitumRouter(
        models=models,
        predictors=cast(Dict[str, Dict[str, CalibratedPredictor]], predictors),
        solver=solver,
        coherence=coherence,
        boundary=boundary,
        srmf=srmf,
        pgd_extractor=pgd_extractor,
        metric_map=cast(Dict[str, SymbolicManifoldMetric], metric_map),
        energy=energy,
        update_stride=1,
    )

    embeddings = np.random.rand(2, 2)
    prompts = ["prompt1", "prompt2"]
    certs = router.batch_route(embeddings, prompts)

    assert len(certs) == 2
    assert energy.batch_compute.call_count == 1
    assert solver.select.call_count == 2
    assert boundary.analyze.call_count == 2
    assert metric_map["m1"].batch_update_spd.call_count == 1
    assert srmf.batch_update.call_count == 1
    assert metric_map["m1"].update_spd.call_count == 0
    assert srmf.update.call_count == 0


def _make_single_model_router(
    num_samples: int, stride: int, enable_metric_update: bool = True
) -> tuple[CompitumRouter, MagicMock, MagicMock]:
    model1 = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    energy = MagicMock()
    energy.batch_compute.return_value = (
        np.full(num_samples, 0.9),
        np.full(num_samples, 0.1),
        [{"distance": -0.5} for _ in range(num_samples)],
    )
    type(energy).beta_d = PropertyMock(return_value=0.5)
    metric = MagicMock(spec=SymbolicManifoldMetric)
    metric.batch_update_spd.return_value = 3.0  # distinct from the 1.0 placeholder
    metric_map = {"m1": metric}
    solver = MagicMock()
    solver.select.return_value = (model1, {"feasible": True})
    boundary = MagicMock()
    boundary.analyze.return_value = {"is_boundary": False}
    srmf = MagicMock()
    srmf.batch_update.return_value = ([], [{} for _ in range(num_samples)])
    predictors = {
        "m1": {
            "quality": MagicMock(spec=CalibratedPredictor),
            "latency": MagicMock(spec=CalibratedPredictor),
            "cost": MagicMock(spec=CalibratedPredictor),
        }
    }
    router = CompitumRouter(
        models=[model1],
        predictors=cast(Dict[str, Dict[str, CalibratedPredictor]], predictors),
        solver=solver,
        coherence=MagicMock(),
        boundary=boundary,
        srmf=srmf,
        pgd_extractor=MagicMock(),
        metric_map=cast(Dict[str, SymbolicManifoldMetric], metric_map),
        energy=energy,
        update_stride=stride,
        enable_metric_update=enable_metric_update,
    )
    return router, metric, srmf


def test_batch_route_step_accumulates_not_resets_or_doubles() -> None:
    """`self._step` must accumulate by exactly 1 per sample (not reset to a
    constant, and not double-increment) -- with stride=4 over 6 samples, the
    per-sample trigger (`step % stride == 0`) should fire exactly once (at
    step 4, the 4th sample), giving `batch_update_spd` a 1-row batch. A reset
    to a constant would make it fire 0 times; a += 2 step would make it fire
    3 times (steps 4, 8, 12); an `and`->`or` on the per-sample gate would
    make it fire on all 6 samples regardless of stride."""
    router, metric, _ = _make_single_model_router(num_samples=6, stride=4)
    embeddings = np.random.rand(6, 2)
    router.batch_route(embeddings, [f"p{i}" for i in range(6)])
    assert metric.batch_update_spd.call_count == 1
    x_batch_arg = metric.batch_update_spd.call_args[0][0]
    assert x_batch_arg.shape[0] == 1


def test_batch_route_final_step_exactly_at_stride_still_triggers_update() -> None:
    """The batch-level gate is `self._step >= self._stride`, checked once
    after the per-sample loop. With num_samples == stride, the final step
    lands exactly on the boundary -- `>=` must still fire here (a `>`
    mutation would incorrectly skip the update this one time, discarding an
    already-populated update_data batch)."""
    router, metric, _ = _make_single_model_router(num_samples=4, stride=4)
    embeddings = np.random.rand(4, 2)
    router.batch_route(embeddings, [f"p{i}" for i in range(4)])
    assert metric.batch_update_spd.call_count == 1


def test_batch_route_grad_norm_placeholder_and_certificate_matching() -> None:
    """`grad_norm_drift_batch` starts at a `1.0` placeholder per sample, then
    gets overwritten with the real computed grad_norm only for certificates
    whose `.model` matches the just-updated `model_name` (`==`, not `!=`).
    With a single always-selected model and enable_metric_update=True, every
    certificate matches, so every placeholder must be overwritten with the
    mocked batch_update_spd return value (3.0), not left at 1.0."""
    router, metric, srmf = _make_single_model_router(num_samples=4, stride=4)
    embeddings = np.random.rand(4, 2)
    router.batch_route(embeddings, [f"p{i}" for i in range(4)])
    grad_norm_arg = srmf.batch_update.call_args[0][1]
    assert list(grad_norm_arg) == [3.0, 3.0, 3.0, 3.0]


def test_batch_route_grad_norm_placeholder_value_when_metric_update_disabled() -> None:
    """With enable_metric_update=False, the metric-update phase never runs,
    so every certificate's grad_norm must remain at the literal `1.0`
    placeholder -- never the metric-update return value, and never a
    different placeholder constant."""
    router, metric, srmf = _make_single_model_router(
        num_samples=3, stride=1, enable_metric_update=False
    )
    embeddings = np.random.rand(3, 2)
    router.batch_route(embeddings, [f"p{i}" for i in range(3)])
    assert metric.batch_update_spd.call_count == 0
    grad_norm_arg = srmf.batch_update.call_args[0][1]
    assert list(grad_norm_arg) == [1.0, 1.0, 1.0]


def test_batch_route_per_model_update_data_uses_continue_not_break() -> None:
    """The per-model batch-update loop must `continue` past a model with no
    accumulated update data, not `break` out of the whole loop -- otherwise
    an earlier, never-selected model (empty data) would silently prevent
    every later model's real update from ever running."""
    model1 = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    model2 = Model(name="m2", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    energy = MagicMock()
    energy.batch_compute.return_value = (
        np.array([0.9, 0.9]),
        np.array([0.1, 0.1]),
        [{"distance": -0.5}, {"distance": -0.5}],
    )
    type(energy).beta_d = PropertyMock(return_value=0.5)
    metric1 = MagicMock(spec=SymbolicManifoldMetric)
    metric2 = MagicMock(spec=SymbolicManifoldMetric)
    metric2.batch_update_spd.return_value = 3.0
    metric_map = {"m1": metric1, "m2": metric2}
    solver = MagicMock()
    # m1 is never selected -- its update_data stays empty; m2 always is.
    solver.select.return_value = (model2, {"feasible": True})
    boundary = MagicMock()
    boundary.analyze.return_value = {"is_boundary": False}
    srmf = MagicMock()
    srmf.batch_update.return_value = ([], [{}, {}])
    predictors = {
        name: {
            "quality": MagicMock(spec=CalibratedPredictor),
            "latency": MagicMock(spec=CalibratedPredictor),
            "cost": MagicMock(spec=CalibratedPredictor),
        }
        for name in ("m1", "m2")
    }
    router = CompitumRouter(
        models=[model1, model2],  # m1 first, matching update_data's iteration order
        predictors=cast(Dict[str, Dict[str, CalibratedPredictor]], predictors),
        solver=solver,
        coherence=MagicMock(),
        boundary=boundary,
        srmf=srmf,
        pgd_extractor=MagicMock(),
        metric_map=cast(Dict[str, SymbolicManifoldMetric], metric_map),
        energy=energy,
        update_stride=1,
    )
    router.batch_route(np.random.rand(2, 2), ["p0", "p1"])
    assert metric2.batch_update_spd.call_count == 1


def test_batch_route_default_prompts_are_empty_strings() -> None:
    """`prompts=None` should default to empty strings per sample, not some
    other placeholder -- observable via the resulting pgd_signature, which
    is a hash of the (empty) prompt string."""
    router, _, _ = _make_single_model_router(num_samples=2, stride=1000)
    certs = router.batch_route(np.random.rand(2, 2))
    assert certs[0].pgd_signature == pgd_hash("")
    assert certs[1].pgd_signature == pgd_hash("")


def test_batch_route_metric_update_eta_is_exact() -> None:
    router, metric, _ = _make_single_model_router(num_samples=4, stride=4)
    router.batch_route(np.random.rand(4, 2), [f"p{i}" for i in range(4)])
    assert metric.batch_update_spd.call_args.kwargs["eta"] == 1e-2


def test_batch_route_disabled_controller_reports_exact_current_state() -> None:
    """Mirrors the single-item route()'s disabled-controller test -- the
    batch path builds its own separate drift_statuses list of dicts, whose
    keys were never checked against the controller's real attribute values,
    only that batch_route runs at all."""
    router, _, srmf = _make_single_model_router(
        num_samples=2, stride=1000, enable_metric_update=False
    )
    router.enable_controller = False
    srmf.trust_radius = 1.23
    srmf.drift_ema = 0.45
    srmf.drift_integral = 0.67
    srmf.lyapunov_function.return_value = 0.89
    certs = router.batch_route(np.random.rand(2, 2), ["p0", "p1"])
    for cert in certs:
        assert cert.drift_status == {
            "trust_radius": 1.23,
            "drift_ema": 0.45,
            "drift_integral": 0.67,
            "lyapunov_function": 0.89,
        }


def test_batch_route_print_elapsed_time_is_bounded() -> None:
    """The existing debug-print test only regex-matches the printed line's
    *structure* (\\d+\\.\\d{4}), which a `time.time() + start_time` mutation
    would still satisfy (a huge epoch-scale number still has 4 decimals) --
    bound the parsed elapsed value to actually catch the sign flip."""
    import io
    import re
    from contextlib import redirect_stdout

    router, _, _ = _make_single_model_router(num_samples=1, stride=1000)
    router._step = 99  # so _step reaches 100 (a print-triggering multiple) after 1 sample
    buf = io.StringIO()
    with redirect_stdout(buf):
        router.batch_route(np.random.rand(1, 2), ["p0"])
    out = buf.getvalue()
    elapsed = float(re.search(r"took (\d+\.\d{4}) seconds", out).group(1))  # type: ignore[union-attr]
    assert elapsed < 5.0
