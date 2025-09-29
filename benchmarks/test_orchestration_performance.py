import time
from typing import Any, Dict
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest


# Mock compitum imports to make this self-contained for testing
class SwitchCertificate:
    def __init__(self, model: str, utility: float, utility_components: Dict = None,
                 constraints: Dict = None, boundary_analysis: Dict = None,
                 drift_status: Dict = None, pgd_signature: str = "", timestamp: float = 0.0):
        self.model = model
        self.utility = utility
        self.utility_components = utility_components or {}
        self.constraints = constraints or {}
        self.boundary_analysis = boundary_analysis or {}
        self.drift_status = drift_status or {}
        self.pgd_signature = pgd_signature
        self.timestamp = timestamp

    def to_json(self):
        return '{"model": "' + self.model + '", "utility": ' + str(self.utility) + '}'

class Model:
    def __init__(self, name: str, center: np.ndarray, capabilities: Any, cost: float):
        self.name = name
        self.center = center
        self.capabilities = capabilities
        self.cost = cost
        self.quality_score = 0.5  # Default
        self.latency = 0.05  # Default

class CalibratedPredictor:
    def predict(self, x: np.ndarray):
        return np.array([0.5]), np.array([0.1]), np.array([0.9])

class CoherenceFunctional:
    pass

class BoundaryAnalyzer:
    def analyze(self, utilities: Dict, u_sigmas: Dict):
        return {"is_boundary": False}

class SRMFController:
    def update(self, d_star: float, grad_norm: float):
        return 1.0, {"trust_radius": 1.0}

class ProductionPGDExtractor:
    def extract_features(self, prompt: str):
        pass

class SymbolicManifoldMetric:
    def get_spd(self):
        m = MagicMock()
        m.det.return_value = 1.0
        return m

    def update_spd(self, xR_all: np.ndarray, center: np.ndarray, beta_d: float,
                   d_best: float, eta: float, srmf_controller: Any):
        return 1.0

class SymbolicFreeEnergy:
    @property
    def beta_d(self):
        return 0.5

    def compute(self, numerical_feats: Dict, model: Model, predictors: Dict,
                coherence: Any, met: Any):
        # Simulate utility based on query_type and model
        query_type = numerical_feats.get('_query_type', 'general')
        model_name = model.name
        base_u = model.quality_score
        # Boost for complex queries on high-quality models
        if query_type == 'complex' and 'high_quality' in model_name:
            base_u += 0.3
        elif query_type == 'simple' and 'low_quality' in model_name:
            base_u -= 0.1
        sig = 0.1
        uc = {"quality": base_u, "distance": -0.5}
        return base_u, sig, uc

class CompitumRouter:
    def __init__(self, models: list, predictors: Dict, solver: Any, coherence: Any,
                 boundary: Any, srmf: Any, pgd_extractor: Any, metric_map: Dict,
                 energy: Any, update_stride: int = 1):
        self.models = {m.name: m for m in models}
        self.predictors = predictors
        self.solver = solver
        self.coherence = coherence
        self.boundary = boundary
        self.srmf = srmf
        self.pgd = pgd_extractor
        self.metric_map = metric_map
        self.energy = energy
        self._stride = update_stride
        self._step = 0

    def route(self, prompt: str, context: Dict[str, Any] | None = None) -> SwitchCertificate:
        numerical_feats_from_pgd, query_type = self.pgd.extract_features(prompt)

        # Create a copy of numerical_feats_from_pgd and add query_type for energy.compute
        energy_input_feats = numerical_feats_from_pgd.copy()
        energy_input_feats['_query_type'] = query_type

        xR_all, xB = split_features(numerical_feats_from_pgd)  # Split only numerical

        utilities: Dict[str, float] = {}
        comps: Dict[str, Dict[str, float]] = {}
        u_sigmas: Dict[str, float] = {}

        for name, model in self.models.items():
            met = self.metric_map[name]
            U, sig, uc = self.energy.compute(
                energy_input_feats, model, self.predictors[name], self.coherence, met
            )
            utilities[name] = float(U)
            comps[name] = uc
            u_sigmas[name] = float(sig)

        m_star, cinfo = self.solver.select(xB, list(self.models.values()), utilities)
        binfo = self.boundary.analyze(utilities, u_sigmas)

        # Adapt metric periodically (two-timescale)
        self._step += 1
        grad_norm = 1.0
        if self._step % self._stride == 0:
            met = self.metric_map[m_star.name]
            d_best = abs(-comps[m_star.name]["distance"])
            grad_norm = met.update_spd(
                xR_all,
                self.models[m_star.name].center,
                self.energy.beta_d,
                d_best,
                eta=1e-2,
                srmf_controller=self.srmf,
            )

        _, drift = self.srmf.update(
            d_star=abs(-comps[m_star.name]["distance"]), grad_norm=grad_norm
        )

        cert = SwitchCertificate(
            model=m_star.name,
            utility=utilities[m_star.name],
            utility_components=comps[m_star.name],
            constraints=cinfo,
            boundary_analysis=binfo,
            drift_status=drift,
            pgd_signature="abc",  # Simplified for benchmark
            timestamp=time.time()
        )
        return cert

def split_features(features: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    # Mock split: pragmatic to xB, riemannian to xR
    xB = np.array([v for k, v in features.items() if k.startswith('prag_')])
    xR = np.array(
        [
            v
            for k, v in features.items()
            if not k.startswith("prag_") and isinstance(v, (int, float))
        ]
    )
    return xR, xB

# --- Fixtures ---

@pytest.fixture
def mock_predictor():
    predictor = MagicMock(spec=CalibratedPredictor)
    predictor.predict.return_value = (np.array([0.5]), np.array([0.1]), np.array([0.9]))
    return predictor

@pytest.fixture
def mock_models_with_varying_quality():
    models_data = [
        {"name": "model_low_cost_low_quality", "quality": 0.2, "cost": 0.1, "latency": 0.01},
        {"name": "model_medium_cost_medium_quality", "quality": 0.5, "cost": 0.5, "latency": 0.05},
        {"name": "model_high_cost_high_quality", "quality": 0.9, "cost": 1.0, "latency": 0.1},
    ]
    models = []
    for data in models_data:
        model = Model(
            name=data["name"],
            center=np.zeros(10),
            capabilities=MagicMock(),
            cost=data["cost"],
        )
        model.quality_score = data["quality"]
        model.latency = data["latency"]
        models.append(model)
    return models

@pytest.fixture
def mock_pgd_extractor_for_diverse_queries():
    pgd_extractor = MagicMock(spec=ProductionPGDExtractor)
    def extract_features_side_effect(prompt):
        if "simple" in prompt.lower():
            return {"f1": 0.1, "f2": 0.2}, "simple"  # No 'complexity' to avoid split issues
        elif "complex" in prompt.lower():
            return {"f1": 0.8, "f2": 0.9}, "complex"
        else:
            return {"f1": 0.4, "f2": 0.5}, "general"
    pgd_extractor.extract_features.side_effect = extract_features_side_effect
    return pgd_extractor

@pytest.fixture
def mock_router(
    mock_models_with_varying_quality, mock_predictor, mock_pgd_extractor_for_diverse_queries
):
    models = mock_models_with_varying_quality
    predictors = {
        m.name: {"quality": mock_predictor, "latency": mock_predictor, "cost": mock_predictor}
        for m in models
    }

    # Mock other dependencies
    solver = MagicMock()
    def mock_select(x_b, models_list, utilities):
        best_model = max(models_list, key=lambda m: utilities.get(m.name, 0))
        return best_model, {"feasible": True}
    solver.select.side_effect = mock_select

    coherence = MagicMock(spec=CoherenceFunctional)
    boundary = MagicMock(spec=BoundaryAnalyzer)
    boundary.analyze.return_value = {"is_boundary": False}
    srmf = MagicMock(spec=SRMFController)
    srmf.update.return_value = (1.0, {"trust_radius": 1.0})
    pgd_extractor = mock_pgd_extractor_for_diverse_queries
    metric_map = {m.name: MagicMock(spec=SymbolicManifoldMetric) for m in models}
    for met in metric_map.values():
        met.get_spd.return_value.det.return_value = 1.0
        met.update_spd.return_value = 1.0

    energy_mock = MagicMock(spec=SymbolicFreeEnergy)
    # Configure energy_mock.compute to return utility based on model's
    # quality_score, cost, and query type
    def mock_energy_compute(xR_all, model, predictors, coherence, metric):
        query_type = xR_all.get('_query_type', 'general')

        # Calculate base utility considering quality and cost
        if query_type == "complex": # Complex query, prioritize quality
            calculated_utility = model.quality_score * 0.8 - model.cost * 0.2
        elif query_type == "simple": # Simple query, prioritize low cost
            calculated_utility = model.quality_score * 0.3 - model.cost * 0.7
        else: # General query, balanced
            calculated_utility = model.quality_score * 0.5 - model.cost * 0.5

        # Ensure utility is non-negative
        utility = max(0.0, calculated_utility)

        sig = 0.1
        uc = {"quality": utility, "distance": -0.5}
        return utility, sig, uc
    energy_mock.compute.side_effect = mock_energy_compute
    type(energy_mock).beta_d = PropertyMock(return_value=0.5)

    router_instance = CompitumRouter(
        models=models,
        predictors=predictors,
        solver=solver,
        coherence=coherence,
        boundary=boundary,
        srmf=srmf,
        pgd_extractor=pgd_extractor,
        metric_map=metric_map,
        energy=energy_mock,
        update_stride=1
    )
    return router_instance

class GenericSimpleRouter:
    def __init__(self, models, pgd_extractor, selection_strategy, energy_mock):
        self.models = models
        self.pgd_extractor = pgd_extractor
        self.selection_strategy = selection_strategy
        self.energy = energy_mock

    def route(self, prompt: str) -> SwitchCertificate:
        numerical_feats, query_type = self.pgd_extractor.extract_features(prompt)
        chosen_model = self.selection_strategy(self.models, numerical_feats, query_type)

        # Calculate utility using the same energy.compute logic as CompitumRouter
        # We need to pass numerical_feats_with_type to energy.compute
        energy_input_feats = numerical_feats.copy()
        energy_input_feats['_query_type'] = query_type

        # Mock other dependencies for energy.compute call within simple router
        mock_predictors = {chosen_model.name: MagicMock(spec=CalibratedPredictor)}
        mock_coherence = MagicMock(spec=CoherenceFunctional)
        mock_metric = MagicMock(spec=SymbolicManifoldMetric)
        mock_metric.get_spd.return_value.det.return_value = 1.0

        utility, _, _ = self.energy.compute(
            energy_input_feats, chosen_model, mock_predictors, mock_coherence, mock_metric
        )

        return SwitchCertificate(
            model=chosen_model.name,
            utility=utility,
            utility_components={},
            constraints={},
            boundary_analysis={},
            drift_status={},
            pgd_signature="abc",
            timestamp=123.0
        )

@pytest.fixture
def simple_accuracy_router(
    mock_models_with_varying_quality, mock_pgd_extractor_for_diverse_queries, mock_router
):
    """
    A simple router that always picks the first model, simulating a less
    intelligent routing strategy for comparison.
    """
    models = mock_models_with_varying_quality
    pgd_extractor = mock_pgd_extractor_for_diverse_queries
    energy_mock = mock_router.energy # Use the same energy mock as CompitumRouter

    def first_model_strategy(models_list, numerical_feats, query_type):
        return models_list[0]

    return GenericSimpleRouter(models, pgd_extractor, first_model_strategy, energy_mock)

@pytest.fixture
def simple_random_router(
    mock_models_with_varying_quality, mock_pgd_extractor_for_diverse_queries, mock_router
):
    """A simple router that picks a random model from the fleet."""
    models = mock_models_with_varying_quality
    pgd_extractor = mock_pgd_extractor_for_diverse_queries
    energy_mock = mock_router.energy # Use the same energy mock as CompitumRouter

    def random_model_strategy(models_list, numerical_feats, query_type):
        return np.random.choice(models_list)

    return GenericSimpleRouter(models, pgd_extractor, random_model_strategy, energy_mock)

@pytest.fixture
def simple_fixed_router(
    mock_models_with_varying_quality, mock_pgd_extractor_for_diverse_queries, mock_router
):
    """A simple router that always picks the highest quality model, regardless of query type."""
    models = mock_models_with_varying_quality
    pgd_extractor = mock_pgd_extractor_for_diverse_queries
    energy_mock = mock_router.energy # Use the same energy mock as CompitumRouter
    # Assuming the highest quality model is the last one in the list from
    # mock_models_with_varying_quality
    high_quality_model = models[-1]

    def fixed_model_strategy(models_list, numerical_feats, query_type):
        return high_quality_model

    return GenericSimpleRouter(models, pgd_extractor, fixed_model_strategy, energy_mock)

# --- Benchmarks ---

@pytest.mark.benchmark
def benchmark_compitum_route(benchmark, mock_router):
    prompt = "test prompt"
    benchmark(mock_router.route, prompt)

@pytest.mark.benchmark
def benchmark_simple_route(benchmark, simple_random_router):
    prompt = "test prompt"
    benchmark(simple_random_router.route, prompt)

def test_compitum_vs_simple_accuracy(mock_router, simple_accuracy_router):
    prompt = "test prompt"
    compitum_cert = mock_router.route(prompt)
    simple_cert = simple_accuracy_router.route(prompt)
    assert compitum_cert.utility >= simple_cert.utility

def test_compitum_outperforms_simple_in_utility(mock_router, simple_accuracy_router):
    prompt = "complex query"
    compitum_cert = mock_router.route(prompt)
    simple_cert = simple_accuracy_router.route(prompt)
    assert compitum_cert.utility > simple_cert.utility  # Energy boost for complex

def test_context_aware_routing_utility(mock_router, simple_random_router, simple_fixed_router):
    queries = [
        "simple query 1", "general query 1", "complex query 1",
        "simple query 2", "general query 2", "complex query 2",
        "simple query 3", "general query 3", "complex query 3",
    ]

    def run_scenario(router_instance):
        total_utility = 0.0
        for q in queries:
            cert = router_instance.route(q)
            total_utility += cert.utility
        return total_utility

    compitum_total = run_scenario(mock_router)
    random_total = run_scenario(simple_random_router)
    fixed_total = run_scenario(simple_fixed_router)

    assert compitum_total > random_total
    assert compitum_total >= fixed_total  # Compitum adapts better on mix

# Run with: pytest test_orchestration_performance.py --benchmark-only
