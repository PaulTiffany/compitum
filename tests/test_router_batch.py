from typing import Dict, cast
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter


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
