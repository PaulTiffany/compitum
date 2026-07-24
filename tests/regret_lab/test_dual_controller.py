"""Online primal-dual reference controller -- update rule and pricing."""

from __future__ import annotations

from compitum.regret_lab.dual_controller import DualController, price_utilities


def test_lambda_starts_at_zero_and_updates_with_positive_error() -> None:
    controller = DualController(resource_names=("budget", "quota"), eta=0.5)
    assert controller.lambda_price == {"budget": 0.0, "quota": 0.0}
    controller.update({"budget": 2.0, "quota": 0.0})
    assert controller.lambda_price["budget"] == 1.0
    assert controller.lambda_price["quota"] == 0.0


def test_lambda_never_goes_negative() -> None:
    controller = DualController(resource_names=("budget",), eta=1.0)
    controller.update({"budget": -100.0})
    assert controller.lambda_price["budget"] == 0.0


def test_lambda_clips_at_lambda_max() -> None:
    controller = DualController(resource_names=("budget",), eta=1.0, lambda_max=5.0)
    controller.update({"budget": 100.0})
    assert controller.lambda_price["budget"] == 5.0


def test_lambda_accumulates_across_updates() -> None:
    controller = DualController(resource_names=("budget",), eta=1.0, lambda_max=100.0)
    controller.update({"budget": 1.0})
    controller.update({"budget": 1.0})
    assert controller.lambda_price["budget"] == 2.0


def test_update_missing_resource_defaults_to_zero_error() -> None:
    controller = DualController(resource_names=("budget", "quota"), eta=1.0)
    controller.update({"budget": 3.0})  # quota omitted
    assert controller.lambda_price["quota"] == 0.0
    assert controller.lambda_price["budget"] == 3.0


def test_price_utilities_subtracts_dot_product_cost() -> None:
    base = {"cheap": 5.0, "rich": 5.0}
    consumption = {
        "cheap": {"budget": 1.0, "quota": 0.0},
        "rich": {"budget": 4.0, "quota": 0.0},
    }
    lambda_price = {"budget": 1.0, "quota": 0.0}
    priced = price_utilities(base, consumption, lambda_price)
    assert priced["cheap"] == 4.0
    assert priced["rich"] == 1.0


def test_price_utilities_with_zero_lambda_equals_base_utility() -> None:
    base = {"a": 3.0}
    consumption = {"a": {"budget": 10.0}}
    priced = price_utilities(base, consumption, {"budget": 0.0})
    assert priced["a"] == 3.0


def test_lambda_price_explicit_constructor_arg_is_respected() -> None:
    controller = DualController(resource_names=("budget",), lambda_price={"budget": 2.5})
    assert controller.lambda_price == {"budget": 2.5}
