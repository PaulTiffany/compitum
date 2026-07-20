import io
import re
from contextlib import redirect_stdout
from typing import Dict, cast
from unittest.mock import MagicMock

import numpy as np

from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter, SwitchCertificate


def test_switch_certificate_to_json_simple() -> None:
    cert = SwitchCertificate(
        model="m1",
        utility=1.234567,
        utility_components={"distance": -0.5},
        constraints={"feasible": True},
        boundary_analysis={},
        drift_status={},
        pgd_signature="abcd1234abcd1234",
        timestamp=0.0,
    )
    s = cert.to_json()
    assert "m1" in s and "abcd1234" in s


def _make_router(update_stride: int = 1) -> CompitumRouter:
    model = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    energy = MagicMock()
    energy.compute.return_value = (0.9, 0.1, {"distance": -0.5})
    energy.batch_compute.return_value = (
        np.array([0.9, 0.95]),
        np.array([0.1, 0.1]),
        [{"distance": -0.5}, {"distance": -0.6}],
    )
    solver = MagicMock()
    solver.select.return_value = (model, {"feasible": True})
    lyapunov_controller = MagicMock()
    lyapunov_controller.update.return_value = (1.0, {"trust_radius": 1.0, "lyapunov_function": 0.0})
    lyapunov_controller.batch_update.return_value = ([], [{}, {}])
    boundary = MagicMock()
    boundary.analyze.return_value = {"is_boundary": False}
    metric = MagicMock(spec=SymbolicManifoldMetric)
    metric.update_spd.return_value = 1.0
    router = CompitumRouter(
        models=[model],
        predictors={
            "m1": {
                "quality": MagicMock(spec=CalibratedPredictor),
                "latency": MagicMock(spec=CalibratedPredictor),
                "cost": MagicMock(spec=CalibratedPredictor),
            }
        },
        solver=solver,
        coherence=MagicMock(),
        boundary=boundary,
        srmf=lyapunov_controller,
        pgd_extractor=MagicMock(),
        metric_map={"m1": metric},
        energy=energy,
        update_stride=update_stride,
    )
    return router


def test_router_route_executes() -> None:
    router = _make_router(update_stride=1)
    cert = router.route("hello world")
    assert isinstance(cert.model, str) and cert.model


def test_router_batch_route_executes_stride_update() -> None:
    router = _make_router(update_stride=1)
    embs = np.random.rand(2, 2)
    certs = router.batch_route(embeddings=embs, prompts=["a", "b"])
    assert len(certs) == 2


def test_router_route_with_embedding_executes() -> None:
    router = _make_router(update_stride=10)
    emb = np.random.rand(4)
    cert = router.route("ignored", embedding=emb)
    assert isinstance(cert.model, str)


def test_router_batch_route_no_stride_update() -> None:
    router = _make_router(update_stride=1000)
    embs = np.random.rand(2, 2)
    certs = router.batch_route(embeddings=embs, prompts=["a", "b"])
    assert len(certs) == 2


def test_router_stride_clamps_to_one() -> None:
    """No existing test passes update_stride <= 0 -- the
    max(int(update_stride), 1) floor that keeps `self._step % self._stride`
    from ever dividing by zero was never exercised."""
    assert _make_router(update_stride=0)._stride == 1
    assert _make_router(update_stride=-5)._stride == 1


def test_router_srmf_is_legacy_alias_for_controller() -> None:
    router = _make_router(update_stride=1)
    assert router.srmf is router.controller


def test_router_disabled_controller_reports_exact_current_state() -> None:
    """The only existing disabled-controller test just checks "trust_radius"
    is a key present in drift_status -- it never checks the actual values,
    so a mutation swapping which controller attribute lands under which key
    would survive. Assert the full exact dict."""
    model = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    energy = MagicMock()
    energy.compute.return_value = (0.9, 0.1, {"distance": -0.5})
    solver = MagicMock()
    solver.select.return_value = (model, {"feasible": True})
    controller = MagicMock()
    controller.trust_radius = 1.23
    controller.drift_ema = 0.45
    controller.drift_integral = 0.67
    controller.lyapunov_function.return_value = 0.89
    boundary = MagicMock()
    boundary.analyze.return_value = {"is_boundary": False}
    metric = MagicMock(spec=SymbolicManifoldMetric)

    router = CompitumRouter(
        models=[model],
        predictors={
            "m1": {
                "quality": MagicMock(spec=CalibratedPredictor),
                "latency": MagicMock(spec=CalibratedPredictor),
                "cost": MagicMock(spec=CalibratedPredictor),
            }
        },
        solver=solver,
        coherence=MagicMock(),
        boundary=boundary,
        srmf=controller,
        pgd_extractor=MagicMock(),
        metric_map={"m1": metric},
        energy=energy,
        update_stride=1,
        enable_metric_update=False,
        enable_controller=False,
    )
    cert = router.route("hello")
    assert cert.drift_status == {
        "trust_radius": 1.23,
        "drift_ema": 0.45,
        "drift_integral": 0.67,
        "lyapunov_function": 0.89,
    }


def test_router_batch_route_debug_print_exact_content() -> None:
    """batch_route()'s debug print used to be `# pragma: no cover`-excluded
    (removed per policy: no pragma shortcuts, only genuine equivalent
    mutants). It also had no working test at all -- the only prior attempts
    were commented-out dead code. Assert the full printed line's structure
    via regex (elapsed time is non-deterministic)."""
    model1 = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    energy = MagicMock()
    energy.batch_compute.return_value = (
        np.array([0.9] * 100),
        np.array([0.1] * 100),
        [{"distance": -0.5}] * 100,
    )
    solver = MagicMock()
    solver.select.return_value = (model1, {"feasible": True})
    srmf = MagicMock()
    srmf.batch_update.return_value = ([1.0] * 100, [{"trust_radius": 1.0}] * 100)
    router = CompitumRouter(
        models=[model1],
        predictors=cast(
            Dict[str, Dict[str, CalibratedPredictor]],
            {
                "m1": {
                    "quality": MagicMock(spec=CalibratedPredictor),
                    "latency": MagicMock(spec=CalibratedPredictor),
                    "cost": MagicMock(spec=CalibratedPredictor),
                },
            },
        ),
        solver=solver,
        coherence=MagicMock(),
        boundary=MagicMock(),
        srmf=srmf,
        pgd_extractor=MagicMock(),
        metric_map=cast(
            Dict[str, SymbolicManifoldMetric], {"m1": MagicMock(spec=SymbolicManifoldMetric)}
        ),
        energy=energy,
        update_stride=1000,
    )
    router._step = 0
    buf = io.StringIO()
    with redirect_stdout(buf):
        router.batch_route(np.random.rand(100, 2), [f"p{i}" for i in range(100)])
    out = buf.getvalue()
    assert re.fullmatch(
        r"CompitumRouter\.batch_route took \d+\.\d{4} seconds for 100 samples\n", out
    )
