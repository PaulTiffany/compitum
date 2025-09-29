import copy
from typing import Any, List, Tuple

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from compitum.capabilities import Capabilities
from compitum.coherence import CoherenceFunctional
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.pgd import ProductionPGDExtractor
from compitum.predictors import CalibratedPredictor
from compitum.utils import split_features

from .harness import TOL

# --- Strategies for generating test data ---


@st.composite
def model_instance(draw: Any, dim: int = 35) -> Model:
    name = draw(st.text(min_size=1, max_size=10))
    center = np.array(draw(st.lists(st.floats(-1.0, 1.0), min_size=dim, max_size=dim)))

    regions = set(draw(st.lists(st.text(min_size=1, max_size=8), max_size=3)))
    tools_allowed = set(draw(st.lists(st.text(min_size=1, max_size=8), max_size=3)))
    deterministic = draw(st.booleans())
    capabilities = Capabilities(
        regions=regions,
        tools_allowed=tools_allowed,
        deterministic=deterministic,
    )

    cost = draw(st.floats(0.1, 10.0))
    return Model(name=name, center=center, capabilities=capabilities, cost=cost)


def dummy_calibrated_predictor() -> SearchStrategy[CalibratedPredictor]:
    """
    Generates a dummy CalibratedPredictor instance.
    Since its internal logic isn't critical for this test, we can use a simple mock.
    """
    class DummyPredictor(CalibratedPredictor):
        def __init__(self) -> None:
            # CalibratedPredictor expects these to be initialized
            self.base = None
            self.iso = None
            self.q05 = None
            self.q95 = None
            self.fitted = False

        def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # Return dummy values matching the expected type
            val = np.array([0.5])
            return val, val - 0.1, val + 0.1

        def update(self, x: np.ndarray, y: float) -> None:
            pass # No-op update

    return st.just(DummyPredictor())


@st.composite
def prompt_strategy(draw: Any) -> str:
    """Generates a string prompt."""
    return draw(st.text(min_size=1, max_size=100))


# --- Invariant Tests ---


@pytest.mark.invariants
@given(
    models=st.lists(model_instance(), min_size=1, max_size=5),
    prompt=prompt_strategy(),
    data=st.data(),
    # Strategies for SymbolicFreeEnergy
    alpha=st.floats(0.1, 1.0),
    beta_t=st.floats(0.1, 1.0),
    beta_c=st.floats(0.1, 1.0),
    beta_d=st.floats(0.1, 1.0),
    beta_s=st.floats(0.1, 1.0),
    # Strategies for ReflectiveConstraintSolver
    num_constraints=st.integers(1, 3),
)
def test_router_cost_inflation_reduces_utility(
    models: List[Model],
    prompt: str,
    data: Any,
    alpha: float,
    beta_t: float,
    beta_c: float,
    beta_d: float,
    beta_s: float,
    num_constraints: int,
) -> None:
    """
    Metamorphic test: Increasing a model's cost should decrease its utility.
    POWER: Metamorphic/Economic
    """
    assume(len(models) > 0)

    # Determine dimension from models
    dim = models[0].center.shape[0]
    rank = min(dim, 2) # Example rank

    # Create dependencies for CompitumRouter
    predictors = {m.name: {"utility": data.draw(dummy_calibrated_predictor()),
                           "latency": data.draw(dummy_calibrated_predictor()),
                           "cost": data.draw(dummy_calibrated_predictor()),
                           "quality": data.draw(dummy_calibrated_predictor())} for m in models}

    coherence = CoherenceFunctional()
    pgd_extractor = ProductionPGDExtractor() # No dim argument
    metric_map = {m.name: SymbolicManifoldMetric(D=dim, rank=rank) for m in models}
    energy = SymbolicFreeEnergy(
        alpha=alpha, beta_t=beta_t, beta_c=beta_c, beta_d=beta_d, beta_s=beta_s
    )

    # 1. Get initial utility of the target model
    models_initial = copy.deepcopy(models)
    model_to_inflate_idx = data.draw(st.integers(0, len(models_initial) - 1))
    model_name = models_initial[model_to_inflate_idx].name

    target_model_initial = next(m for m in models_initial if m.name == model_name)
    feats_initial = pgd_extractor.extract_features(prompt)
    xR_all_initial, _ = split_features(feats_initial)
    initial_utility_target, _, _ = energy.compute(
        xR_all_initial, target_model_initial, predictors[model_name],
        coherence, metric_map[model_name]
    )

    # 2. Metamorphic Change: Inflate a model's cost
    models_new = copy.deepcopy(models)
    # Find the model to inflate in the new list
    target_model_new = next(m for m in models_new if m.name == model_name)
    inflation_factor = data.draw(st.floats(1.1, 5.0))
    target_model_new.cost *= inflation_factor

    # Re-use dependencies that do not rely on the changed model attributes
    predictors_new = predictors
    metric_map_new = metric_map

    # 3. Get new utility of the target model
    feats_new = pgd_extractor.extract_features(prompt)
    xR_all_new, _ = split_features(feats_new)
    new_utility_target, _, _ = energy.compute(
        xR_all_new, target_model_new, predictors_new[model_name],
        coherence, metric_map_new[model_name]
    )

    # 4. Assertion
    assert new_utility_target <= initial_utility_target + TOL.abs
