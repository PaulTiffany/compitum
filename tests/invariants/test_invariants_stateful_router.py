from typing import Any, Dict, cast

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import SRMFController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.pgd import ProductionPGDExtractor
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter

from .harness import TOL
from .test_invariants_router import (
    dummy_calibrated_predictor,
    model_instance,
    prompt_strategy,
)


@st.composite
def routers(draw: Any) -> CompitumRouter:
    dim = 35
    rank = draw(st.integers(1, 5))
    models = draw(st.lists(model_instance(dim=dim), min_size=1, max_size=3))

    predictors = {
        m.name: {
            "quality": draw(dummy_calibrated_predictor()),
            "latency": draw(dummy_calibrated_predictor()),
            "cost": draw(dummy_calibrated_predictor()),
        }
        for m in models
    }

    num_constraints = draw(st.integers(1, 2))
    banach_dim = 4
    A = np.array(
        draw(
            st.lists(
                st.lists(st.floats(-1, 1), min_size=banach_dim, max_size=banach_dim),
                min_size=num_constraints,
                max_size=num_constraints,
            )
        )
    )
    b = np.array(
        draw(st.lists(st.floats(-10, 10), min_size=num_constraints, max_size=num_constraints))
    )

    solver = ReflectiveConstraintSolver(A, b)
    coherence = CoherenceFunctional()
    boundary = BoundaryAnalyzer()
    srmf = SRMFController()
    pgd = ProductionPGDExtractor()
    metrics = {m.name: SymbolicManifoldMetric(D=dim, rank=rank) for m in models}
    energy = draw(
        st.builds(
            SymbolicFreeEnergy,
            alpha=st.floats(0.1, 1.0),
            beta_t=st.floats(0.1, 1.0),
            beta_c=st.floats(0.1, 1.0),
            beta_d=st.floats(0.1, 1.0),
            beta_s=st.floats(0.1, 1.0),
        )
    )

    return CompitumRouter(
        models,
        cast(Dict[str, Dict[str, CalibratedPredictor]], predictors),
        solver,
        coherence,
        boundary,
        srmf,
        pgd,
        metrics,
        energy,
    )


class RouterLifecycle(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.router: CompitumRouter | None = None

    @rule(router=routers())
    def initialize_router(self, router: CompitumRouter) -> None:
        self.router = router

    @rule(prompt=prompt_strategy())
    @precondition(lambda self: self.router is not None)
    def route_prompt(self, prompt: str) -> None:
        assert self.router is not None
        cert = self.router.route(prompt)
        assert cert is not None
        assert cert.model in self.router.models

    @invariant()
    @precondition(lambda self: self.router is not None)
    def metrics_are_spd(self) -> None:
        assert self.router is not None
        for name, metric in self.router.metric_map.items():
            M = metric.metric_matrix()
            assert np.allclose(M, M.T, atol=TOL.abs), f"Metric for {name} not symmetric"
            try:
                eigvals = np.linalg.eigvalsh(M)
                assert np.all(eigvals > 0), f"Metric for {name} not positive-definite"
            except np.linalg.LinAlgError:
                pytest.fail(f"Metric for {name} not positive definite, leading to LinAlgError")

    @invariant()
    @precondition(lambda self: self.router is not None)
    def controller_is_stable(self) -> None:
        assert self.router is not None
        r = self.router.srmf.r
        assert 0.2 <= r <= 5.0, "SRMF trust radius out of bounds"


TestRouterLifecycle = RouterLifecycle.TestCase
