
import numpy as np

from compitum.capabilities import Capabilities
from compitum.models import Model


def test_model_creation() -> None:
    caps = Capabilities(regions={"US"}, tools_allowed={"none"})
    model = Model(
        name="test_model",
        center=np.array([1.0, 2.0]),
        capabilities=caps,
        cost=0.0
    )
    assert model.name == "test_model"
    np.testing.assert_array_equal(model.center, np.array([1.0, 2.0]))
    assert model.capabilities == caps
