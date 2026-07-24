"""Two-part model, metrics, and feature-extraction helpers -- all
dependency-free, tested with synthetic (but genuinely learnable) data so
correctness is verified, not just "runs without crashing"."""

from __future__ import annotations

import numpy as np
import pytest

from compitum.constraint_oracle.experiment import (
    calibrate_threshold,
    classification_metrics,
    fit_two_part_model,
    predict_two_part,
    ranking_accuracy,
    regression_metrics,
    shuffle_raw_steps,
    stratify_by_threshold,
    terminal_features_from_evidence,
    trajectory_features_from_evidence,
)


def _synthetic_dataset(n: int, seed: int):
    """x0 alone determines both consequential-ness (x0>0) and, when
    consequential, the magnitude (proportional to x0) -- a model that
    actually uses x0 should score well; one that ignores it should not."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    consequential = X[:, 0] > 0
    magnitude = [float(2.0 * X[i, 0]) if consequential[i] else None for i in range(n)]
    return X, list(consequential), magnitude


def test_two_part_model_learns_classification_and_regression() -> None:
    X_train, cons_train, mag_train = _synthetic_dataset(200, seed=1)
    X_test, cons_test, mag_test = _synthetic_dataset(100, seed=2)

    model = fit_two_part_model(X_train, cons_train, mag_train)
    assert model.regressor_fitted is True

    p_pred, magnitude_pred = predict_two_part(model, X_test)
    cls_metrics = classification_metrics(cons_test, p_pred)
    assert cls_metrics["accuracy"] > 0.8

    rows = [i for i, c in enumerate(cons_test) if c]
    reg_metrics = regression_metrics([mag_test[i] for i in rows], [magnitude_pred[i] for i in rows])
    assert reg_metrics["mae"] < 1.0


def test_two_part_model_with_no_consequential_rows_skips_regressor() -> None:
    X = np.random.default_rng(3).normal(size=(20, 3))
    consequential = [False] * 20
    magnitude: list = [None] * 20
    model = fit_two_part_model(X, consequential, magnitude)
    assert model.regressor_fitted is False
    _, magnitude_pred = predict_two_part(model, X)
    assert np.all(magnitude_pred == 0.0)


def test_mismatched_lengths_are_rejected() -> None:
    X = np.zeros((3, 2))
    with pytest.raises(ValueError, match="align"):
        fit_two_part_model(X, [True, False], [1.0, None])


def test_classification_metrics_perfect_and_empty() -> None:
    perfect = classification_metrics([True, False, True], [1.0, 0.0, 1.0])
    assert perfect["accuracy"] == 1.0
    assert perfect["precision"] == 1.0
    assert perfect["recall"] == 1.0
    assert perfect["brier_score"] == 0.0

    empty = classification_metrics([], [])
    assert empty["n"] == 0.0
    assert np.isnan(empty["accuracy"])


def test_classification_metrics_no_positive_predictions() -> None:
    metrics = classification_metrics([True, True], [0.1, 0.2])
    assert metrics["recall"] == 0.0
    assert np.isnan(metrics["precision"])  # no positive predictions at all


def test_regression_metrics_basic() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] > metrics["mae"]  # RMSE penalizes the one large error more

    empty = regression_metrics([], [])
    assert empty["n"] == 0.0
    assert np.isnan(empty["mae"])


def test_ranking_accuracy_perfect_and_ties() -> None:
    cases = [
        [(1.0, 0.9), (5.0, 5.1), (0.0, 0.1)],  # true argmax=1, pred argmax=1: correct
        [(3.0, 0.5), (1.0, 9.0), (0.0, 0.1)],  # true argmax=0, pred argmax=1: wrong
        [(1.0, 1.0), (1.0, 1.0)],  # all-tied true values: skipped
    ]
    accuracy = ranking_accuracy(cases)
    assert accuracy == pytest.approx(0.5)  # 1 correct out of 2 scored (tie case skipped)


def test_ranking_accuracy_all_tied_is_nan() -> None:
    cases = [[(1.0, 1.0), (1.0, 2.0)]]
    assert np.isnan(ranking_accuracy(cases))


def test_stratify_by_threshold() -> None:
    labels = stratify_by_threshold([-1.0, 0.5, 1.5], thresholds=[0.0, 1.0])
    assert labels == ["<= 0", "> 0", "> 1"]


def _evidence_fixture(energies, grad_total=1.0, error_total=0.5):
    return {
        "terminal": {
            "total_energy": energies[-1],
            "total_latent_grad_norm": grad_total,
            "total_error_norm": error_total,
        },
        "convergence": {
            "terminal_total_energy": energies[-1],
            "energy_reduction_ratio": energies[-1] / energies[0] if energies[0] else -1.0,
            "monotone_decreasing_fraction": 1.0,
            "terminal_latent_grad_norm_total": grad_total,
        },
        "energy_trajectory": energies,
        "per_node": {
            "hidden": {"terminal_energy": energies[-1] * 0.6},
            "latent": {"terminal_energy": energies[-1] * 0.4},
        },
    }


def test_terminal_features_from_evidence() -> None:
    evidence = _evidence_fixture([2.0, 1.0, 0.5])
    features = terminal_features_from_evidence(evidence)
    assert features == [0.5, 1.0, 0.5]


def test_trajectory_features_from_evidence() -> None:
    evidence = _evidence_fixture([2.0, 1.0, 0.5])
    features = trajectory_features_from_evidence(evidence)
    assert features[0] == pytest.approx(0.5)  # terminal_total_energy
    assert features[4] == pytest.approx(1.0)  # first_drop = 2.0 - 1.0
    assert features[5] == pytest.approx(0.3)  # hidden terminal (sorted: hidden, latent)
    assert features[6] == pytest.approx(0.2)  # latent terminal


def test_trajectory_features_single_step_has_zero_first_drop() -> None:
    evidence = _evidence_fixture([1.0])
    features = trajectory_features_from_evidence(evidence)
    assert features[4] == 0.0


def test_shuffle_raw_steps_preserves_values_reorders_and_renames() -> None:
    payload = {"run_id": "abc", "steps": [{"e": 1}, {"e": 2}, {"e": 3}, {"e": 4}, {"e": 5}]}
    shuffled = shuffle_raw_steps(payload, seed=7)
    assert shuffled["run_id"] == "abc-shuffled7"
    assert sorted(s["e"] for s in shuffled["steps"]) == [1, 2, 3, 4, 5]
    assert payload["steps"] == [{"e": 1}, {"e": 2}, {"e": 3}, {"e": 4}, {"e": 5}]  # untouched


def test_shuffle_raw_steps_missing_run_id_defaults() -> None:
    payload = {"steps": [{"e": 1}, {"e": 2}]}
    shuffled = shuffle_raw_steps(payload, seed=1)
    assert shuffled["run_id"] == "unknown-shuffled1"


def test_calibrate_threshold_matches_training_base_rate() -> None:
    p_train = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.9, 0.95]
    y_train = [False] * 8 + [True] * 2  # 20% positive rate
    threshold = calibrate_threshold(p_train, y_train)
    predicted_positive = sum(1 for p in p_train if p >= threshold)
    assert predicted_positive == pytest.approx(2, abs=1)


def test_calibrate_threshold_low_base_rate_finds_a_usable_cutoff() -> None:
    """Reproduces the pilot's real finding: scores that never reach 0.5 but
    do separate the classes should still get a usable, lower threshold."""
    rng = np.random.default_rng(0)
    n = 500
    positive_rate = 0.08
    y_train = rng.random(n) < positive_rate
    p_train = np.where(y_train, rng.uniform(0.1, 0.4, n), rng.uniform(0.0, 0.2, n))
    threshold = calibrate_threshold(p_train, y_train)
    assert threshold < 0.5
    predicted_positive_rate = float(np.mean(p_train >= threshold))
    assert predicted_positive_rate == pytest.approx(positive_rate, abs=0.05)


def test_calibrate_threshold_empty_inputs() -> None:
    assert calibrate_threshold([], []) == 0.5


def test_calibrate_threshold_all_negative_or_all_positive() -> None:
    assert calibrate_threshold([0.1, 0.2, 0.3], [False, False, False]) >= 0.3
    assert calibrate_threshold([0.1, 0.2, 0.3], [True, True, True]) <= 0.1
