from typing import Any, Dict, cast

import numpy as np
import pytest

try:
    from hypothesis import strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
except Exception:
    pytest.skip("hypothesis not installed", allow_module_level=True)

from compitum.boundary import BoundaryAnalyzer
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import LyapunovController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.pgd import RegexPromptExtractor
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
    srmf = LyapunovController()
    pgd = RegexPromptExtractor()
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
        self._last_whitened_counts: Dict[str, int] = {}
        self._last_certificate: Dict[str, Any] | None = None

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
        # Certificate fields sanity
        u_sum = sum(cert.utility_components.values())
        assert np.isfinite(cert.utility) and np.isfinite(u_sum)
        # PGD signature is a SHA-256 hex string (full length here, truncated only on to_json)
        assert isinstance(cert.pgd_signature, str) and len(cert.pgd_signature) == 64
        # Stash for invariants that may read the latest cert
        self._last_certificate = {
            "boundary": cert.boundary_analysis,
            "constraints": cert.constraints,
            "drift": cert.drift_status,
        }
        # Track whitened residual queue sizes per metric
        for name, met in self.router.metric_map.items():
            self._last_whitened_counts[name] = len(met.whitened_residuals)

    @rule()
    @precondition(lambda self: self.router is not None)
    def batch_step(self) -> None:
        """Exercise batch routing and metric update pruning invariants."""
        assert self.router is not None
        # Infer embedding dimension from any metric
        any_name = next(iter(self.router.metric_map))
        D = self.router.metric_map[any_name].D
        n = 1  # Use size 1 to be compatible with dummy predictors in invariants
        embs = np.zeros((n, D))
        certs = self.router.batch_route(embs)
        assert len(certs) == n
        for c in certs:
            assert c.model in self.router.models
            # Check fields presence
            assert isinstance(c.boundary_analysis, dict)
            assert isinstance(c.constraints, dict)
            assert isinstance(c.drift_status, dict)
        # After batch updates, whitened residuals must be trimmed to <= 100
        for met in self.router.metric_map.values():
            assert len(met.whitened_residuals) <= 100

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
        trust_radius = self.router.srmf.trust_radius
        assert 0.2 <= trust_radius <= 5.0, "LyapunovController trust radius out of bounds"

    @invariant()
    @precondition(lambda self: self._last_certificate is not None)
    def certificate_has_boundary_and_constraints(self) -> None:
        assert self._last_certificate is not None
        b = self._last_certificate["boundary"]
        c = self._last_certificate["constraints"]
        # Boundary keys exist and types are correct; handle single-model case
        if b.get("reason") == "insufficient_models":
            assert b.get("is_boundary") is False
        else:
            for key in ("utility_gap", "entropy", "uncertainty", "is_boundary"):
                assert key in b
            assert isinstance(b["is_boundary"], bool)
        # Constraints dict contains required keys
        for key in ("status", "feasible", "shadow_prices"):
            assert key in c
        assert isinstance(c["feasible"], bool)


TestRouterLifecycle = RouterLifecycle.TestCase
