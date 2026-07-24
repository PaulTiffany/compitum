"""Dependency-free experiment core for tranche 2: two-part model fitting,
metrics, and feature extraction from already-materialized trajectory
evidence.

Deliberately separated from ``experiments/fabricpc/tranche2/`` (the JAX
-dependent orchestration): everything here is pure Python/numpy so it can
be fully unit-tested -- including the feature-extraction functions, which
operate on already-computed ``TrajectoryEvidence``-shaped dicts (real or
synthetic fixtures) and never import FabricPC or JAX themselves.

Two-part model, per the tranche 2 pre-registration: (1) a classifier
estimating P(consequential) -- whether a constraint's ``critical_relaxation``
is defined at all; (2) a regressor estimating the magnitude, fit only on
consequential rows and evaluated only there. A large "not consequential"
class must not manufacture a misleadingly good average error by folding
into one plain regression.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

RIDGE_LAMBDA = 1.0


@dataclass
class TwoPartModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    classifier_weights: np.ndarray
    classifier_bias: float
    regressor_weights: Optional[np.ndarray]
    regressor_bias: float
    regressor_fitted: bool


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def _ridge_fit(
    X: np.ndarray, y: np.ndarray, ridge: float = RIDGE_LAMBDA
) -> Tuple[np.ndarray, float]:
    y_mean = float(y.mean())
    A = X.T @ X + ridge * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ (y - y_mean))
    return w, y_mean


def fit_two_part_model(
    X_train: np.ndarray,
    consequential_train: Sequence[bool],
    magnitude_train: Sequence[Optional[float]],
) -> TwoPartModel:
    """Fit the classifier on all training rows and the regressor only on
    rows where ``consequential_train`` is true and a magnitude is present.
    """
    if X_train.shape[0] != len(consequential_train) or X_train.shape[0] != len(magnitude_train):
        raise ValueError("X_train, consequential_train, magnitude_train must align")
    mean, scale = _standardize_fit(X_train)
    Xn = (X_train - mean) / scale

    y_class = np.array([1.0 if c else 0.0 for c in consequential_train])
    class_w, class_bias = _ridge_fit(Xn, y_class)

    rows = [
        i
        for i, (c, m) in enumerate(zip(consequential_train, magnitude_train))
        if c and m is not None
    ]
    if len(rows) >= 2:
        Xr = Xn[rows]
        yr = np.array([float(magnitude_train[i]) for i in rows])  # type: ignore[arg-type]
        reg_w, reg_bias = _ridge_fit(Xr, yr)
        return TwoPartModel(mean, scale, class_w, class_bias, reg_w, reg_bias, True)
    return TwoPartModel(mean, scale, class_w, class_bias, None, 0.0, False)


def predict_two_part(model: TwoPartModel, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p_consequential in [0,1], predicted_magnitude) for every row.
    ``predicted_magnitude`` is defined for every row (needed for ranking
    across constraints within a case) but should only be scored against
    ground truth on rows that are actually consequential."""
    Xn = (X - model.feature_mean) / model.feature_scale
    raw_score = Xn @ model.classifier_weights + model.classifier_bias
    p = np.clip(raw_score, 0.0, 1.0)
    if model.regressor_fitted and model.regressor_weights is not None:
        magnitude = Xn @ model.regressor_weights + model.regressor_bias
    else:
        magnitude = np.zeros(X.shape[0])
    return p, magnitude


def classification_metrics(
    y_true: Sequence[bool], p_pred: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    y = np.array([1.0 if v else 0.0 for v in y_true])
    p = np.clip(np.array(p_pred, dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "brier_score": float("nan"),
            "n": 0.0,
        }
    pred = (p >= threshold).astype(float)
    tp = float(np.sum((pred == 1.0) & (y == 1.0)))
    fp = float(np.sum((pred == 1.0) & (y == 0.0)))
    fn = float(np.sum((pred == 0.0) & (y == 1.0)))
    accuracy = float(np.mean(pred == y))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    brier = float(np.mean((p - y) ** 2))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "brier_score": brier,
        "n": float(len(y)),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0.0}
    y = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(math.sqrt(np.mean((y - p) ** 2)))
    return {"mae": mae, "rmse": rmse, "n": float(len(y))}


def ranking_accuracy(cases: Sequence[Sequence[Tuple[float, float]]]) -> float:
    """``cases`` is one sequence per routing case, each a list of
    (true_significance, predicted_significance) pairs, one per constraint.
    Returns the fraction of cases where argmax(predicted) == argmax(true),
    skipping cases where every true significance is exactly equal (no
    genuine ranking question)."""
    scored = 0
    correct = 0
    for constraints in cases:
        true_values = [t for t, _ in constraints]
        if len(set(true_values)) <= 1:
            continue
        true_best = int(np.argmax(true_values))
        pred_best = int(np.argmax([p for _, p in constraints]))
        scored += 1
        correct += int(true_best == pred_best)
    return correct / scored if scored else float("nan")


def stratify_by_threshold(values: Sequence[float], thresholds: Sequence[float]) -> List[str]:
    """Bucket labels for stratified reporting, e.g. by slack or utility gap."""
    labels = []
    for v in values:
        label = f"<= {thresholds[0]:g}"
        for t in thresholds:
            if v > t:
                label = f"> {t:g}"
        labels.append(label)
    return labels


def terminal_features_from_evidence(evidence: Dict[str, Any]) -> List[float]:
    terminal = evidence["terminal"]
    return [
        terminal.get("total_energy", 0.0),
        terminal.get("total_latent_grad_norm", 0.0),
        terminal.get("total_error_norm", 0.0),
    ]


def trajectory_features_from_evidence(evidence: Dict[str, Any]) -> List[float]:
    convergence = evidence["convergence"]
    energy = evidence["energy_trajectory"]
    first_drop = energy[0] - energy[1] if len(energy) > 1 else 0.0
    per_node = evidence["per_node"]
    node_terminals = [per_node[name]["terminal_energy"] for name in sorted(per_node)]
    return [
        convergence["terminal_total_energy"],
        convergence["energy_reduction_ratio"],
        convergence["monotone_decreasing_fraction"],
        convergence["terminal_latent_grad_norm_total"],
        first_drop,
        *node_terminals,
    ]


def shuffle_raw_steps(payload: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Negative control: destroy temporal order while preserving each step's
    exact per-node values -- same technique as tranche 1's
    ``shuffled_control_payload``, generalized here to any node/step shape."""
    shuffled = {**payload, "steps": list(payload["steps"])}
    rng = random.Random(seed)
    rng.shuffle(shuffled["steps"])
    shuffled["run_id"] = f"{payload.get('run_id', 'unknown')}-shuffled{seed}"
    return shuffled
