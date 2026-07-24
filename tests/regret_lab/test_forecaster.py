"""EWMA consumption forecaster -- learns systematic bias for the chosen
model only, never for models it never observes."""

from __future__ import annotations

import pytest

from compitum.regret_lab.forecaster import EWMAForecaster


def test_predict_with_no_history_returns_expected_unchanged() -> None:
    forecaster = EWMAForecaster()
    expected = {"a": {"budget": 1.0, "quota": 2.0}}
    predicted = forecaster.predict(expected)
    assert predicted == expected
    assert predicted is not expected  # must not alias the input


def test_update_learns_systematic_positive_bias() -> None:
    forecaster = EWMAForecaster(alpha=0.5)
    expected = {"budget": 1.0}
    realized = {"budget": 1.5}
    for _ in range(20):
        forecaster.update("a", expected, realized)
    predicted = forecaster.predict({"a": {"budget": 1.0}})
    assert predicted["a"]["budget"] == pytest.approx(1.5, abs=1e-4)


def test_update_only_affects_the_chosen_model() -> None:
    forecaster = EWMAForecaster(alpha=0.5)
    for _ in range(20):
        forecaster.update("chosen", {"budget": 1.0}, {"budget": 2.0})
    predicted = forecaster.predict({"chosen": {"budget": 1.0}, "never_chosen": {"budget": 1.0}})
    assert predicted["chosen"]["budget"] > 1.5
    assert predicted["never_chosen"]["budget"] == 1.0


def test_callable_interface_matches_predict() -> None:
    forecaster = EWMAForecaster()
    forecaster.update("a", {"budget": 1.0}, {"budget": 1.2})
    expected = {"a": {"budget": 1.0}}
    assert forecaster(expected) == forecaster.predict(expected)


def test_bias_decays_toward_new_residuals() -> None:
    forecaster = EWMAForecaster(alpha=0.5)
    forecaster.update("a", {"budget": 1.0}, {"budget": 2.0})  # residual +1.0
    first = forecaster.predict({"a": {"budget": 1.0}})["a"]["budget"]
    forecaster.update("a", {"budget": 1.0}, {"budget": 1.0})  # residual 0.0
    second = forecaster.predict({"a": {"budget": 1.0}})["a"]["budget"]
    assert second < first
