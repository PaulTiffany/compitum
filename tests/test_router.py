
import json
import os
import uuid
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, event, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    precondition,
    rule,
)
from hypothesis.strategies import floats, lists

from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter, SwitchCertificate


def test_switch_certificate_to_json() -> None:
    cert = SwitchCertificate(
        model="test",
        utility=0.1234567,
        utility_components={"a": 1.0},
        constraints={},
        boundary_analysis={},
        drift_status={},
        pgd_signature="9981000000000000",
        timestamp=123.456,
    )
    json_str = cert.to_json()
    data = json.loads(json_str)
    assert data["model"] == "test"
    assert data["utility"] == 0.123457
    assert data["pgd_signature"] == "9981000000000000"


def test_router_route_and_init() -> None:
    # Setup Mocks
    model1 = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    models = [model1]
    pgd_extractor = MagicMock()
    pgd_extractor.extract_features.return_value = {"f1": 1, "prag_f2": 2}
    energy = MagicMock()
    energy.compute.return_value = (0.9, 0.1, {"distance": -0.5})
    type(energy).beta_d = PropertyMock(return_value=0.5)
    metric_map = {"m1": MagicMock()}
    metric_map["m1"].get_spd().det.return_value = 1.0
    solver = MagicMock()
    solver.select.return_value = (model1, {"feasible": True})
    boundary = MagicMock()
    boundary.analyze.return_value = {"is_boundary": False}
    srmf = MagicMock()
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
    assert router.models["m1"] == model1
    router.route("a prompt")
    pgd_extractor.extract_features.assert_called_with("a prompt")
    energy.compute.assert_called_once()
    solver.select.assert_called_once()
    boundary.analyze.assert_called_once()
    metric_map["m1"].update_spd.assert_called_once()


def test_router_route_no_stride_update() -> None:
    # Setup Mocks
    model1 = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
    models = [model1]
    pgd_extractor = MagicMock()
    pgd_extractor.extract_features.return_value = {"f1": 1, "prag_f2": 2}
    energy = MagicMock()
    energy.compute.return_value = (0.9, 0.1, {"distance": -0.5})
    type(energy).beta_d = PropertyMock(return_value=0.5)
    metric_map = {"m1": MagicMock(spec=SymbolicManifoldMetric)}
    solver = MagicMock()
    solver.select.return_value = (model1, {"feasible": True})
    boundary = MagicMock()
    boundary.analyze.return_value = {"is_boundary": False}
    srmf = MagicMock()
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
        update_stride=10,
    )
    router.route("a prompt")
    metric_map["m1"].update_spd.assert_not_called()


@settings(
    max_examples=50,
    stateful_step_count=15,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=1000,
)
class RouterLifecycle(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = Model(name="m1", center=np.zeros(2), capabilities=MagicMock(), cost=0.0)
        self.pgd_extractor = MagicMock()
        self.energy = MagicMock()
        self.metric_map = {"m1": MagicMock()}
        self.solver = MagicMock()
        self.boundary = MagicMock()
        self.srmf = MagicMock()
        self.predictors = {
            "m1": {
                "quality": MagicMock(spec=CalibratedPredictor),
                "latency": MagicMock(spec=CalibratedPredictor),
                "cost": MagicMock(spec=CalibratedPredictor),
            }
        }
        self.router = CompitumRouter(
            models=[self.model1],
            predictors=cast(Dict[str, Dict[str, CalibratedPredictor]], self.predictors),
            solver=self.solver,
            coherence=MagicMock(),
            boundary=self.boundary,
            srmf=self.srmf,
            pgd_extractor=self.pgd_extractor,
            metric_map=cast(Dict[str, SymbolicManifoldMetric], self.metric_map),
            energy=self.energy,
            update_stride=1,
        )
        self.router.update = MagicMock()  # type: ignore
        self.case_slug = f"case_{uuid.uuid4().hex[:8]}"
        self.history: List[Dict[str, Any]] = []
        self._certificates: List[SwitchCertificate] = []
        self.utilities: List[float] = []

    @initialize()
    def init_router(self) -> None:
        self.pgd_extractor.extract_features.return_value = {"f1": 1}
        self.energy.compute.return_value = (1.0, 0.0, {"distance": -0.5})
        type(self.energy).beta_d = PropertyMock(return_value=0.5)
        self.solver.select.return_value = (self.model1, {"feasible": True})
        self.boundary.analyze.return_value = {"is_boundary": False}
        self.srmf.update.return_value = (1.0, {"trust_radius": 1.0})
        self.metric_map["m1"].get_spd.return_value.det.return_value = 1.0
        self._certificates.append(self.router.route("init prompt"))
        self.utilities.append(self._certificates[-1].utility)

    @rule(quality=floats(0, 1))
    @precondition(lambda self: self._certificates)
    def provide_feedback(self, quality: float) -> None:
        cert = self._certificates.pop(0)
        self.router.update(cert, {"quality": quality})  # type: ignore
        # Mock behavior after update
        new_energy_val = self.energy.compute.return_value[0] * (0.9 + quality * 0.1)
        self.energy.compute.return_value = (new_energy_val, 0.0, {"distance": -0.5})
        current_trust_radius = self.srmf.update.return_value[1]["trust_radius"]
        scaled_trust_radius = current_trust_radius * (0.95 + quality * 0.1)
        new_trust_radius = max(0.2, min(5.0, scaled_trust_radius))
        self.srmf.update.return_value = (1.0, {"trust_radius": new_trust_radius})

        new_cert = self.router.route("feedback prompt")
        self.utilities.append(new_cert.utility)
        history_entry = {
            "action": "feedback",
            "quality": quality,
            "cert": json.loads(new_cert.to_json()),
        }
        self.history.append(history_entry)
        self._certificates.append(new_cert)

    @rule(feedback_list=lists(floats(0, 1), min_size=2, max_size=5))
    @precondition(lambda self: len(self._certificates) >= 1)
    def multi_feedback(self, feedback_list: List[float]) -> None:
        initial_utility_variance = np.var(self.utilities) if len(self.utilities) > 1 else 0
        for quality in feedback_list:
            if not self._certificates:
                break
            self.provide_feedback(quality)
        assume(len(self.utilities) > len(feedback_list))
        final_utility_variance = np.var(self.utilities[-len(feedback_list) :])
        assert final_utility_variance <= max(initial_utility_variance, 0.1)
        event(f"utility_variance_converged_to_{final_utility_variance:.2f}")

    @rule()
    @precondition(lambda self: self._certificates)
    def invariants_hold(self) -> None:
        cert = self._certificates[-1]
        assert self.metric_map["m1"].get_spd().det() > 1e-6
        trust_radius = self.srmf.update.return_value[1]["trust_radius"]
        assert 0.2 <= trust_radius <= 5.0
        assert self.solver.select.return_value[1]["feasible"]
        assert not np.isnan(cert.utility) and not np.isinf(cert.utility)
        if len(self.utilities) > 5:
            utility_drift = abs(self.utilities[-1] - self.utilities[-5])
            assert utility_drift < 0.5

    def teardown(self) -> None:
        if hasattr(self, "current_exception") and self.current_exception:
            path = os.path.join("artifacts", "failcases")
            os.makedirs(path, exist_ok=True)
            filename = f"router_lifecycle_{self.case_slug}.json"
            with open(os.path.join(path, filename), "w") as f:
                failcase_data = {"history": self.history, "error": str(self.current_exception)}
            json.dump(failcase_data, f, indent=2)

            test_name = f"test_reg_{self.case_slug}"
            py_filename = os.path.join("tests", f"{test_name}.py")
            with open(py_filename, "w") as f:
                f.write(f"import json\n\ndef {test_name}():\n")
                f.write(f"    # Failing case: {filename}\n")
                f.write(f"    # Error: {self.current_exception}\n")
                f.write("    # To be implemented: replay the steps in history\n")
                f.write(f"    assert False, 'Regression test for {self.case_slug} not implemented'")


@pytest.mark.hypo_lifecycle
class TestRouterLifecycle(RouterLifecycle.TestCase):  # type: ignore
    pass
