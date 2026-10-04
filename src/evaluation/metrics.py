"""Threshold-free and operating-point metrics for one split.

Utility uses the challenge definition in ``src.evaluation.utility``. The
threshold that maximises utility is chosen by the caller on validation rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from src.evaluation.utility import normalized_utility
from src.labels.targets import reconstructed_onset_index


def binary_rates(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y = np.asarray(labels).astype(int)
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    f1 = (
        float(2 * precision * sensitivity / (precision + sensitivity))
        if precision + sensitivity
        else float("nan")
    )
    return {
        "threshold": float(threshold),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def ranking_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probabilities, dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted": float(p.mean()),
        "positive_hour_rate": float(y.mean()),
        "n_hours": int(y.size),
        "n_positive_hours": int(y.sum()),
    }


def calibration_slope_intercept(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    y = np.asarray(labels).astype(int)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    estimator = LogisticRegression(C=1e12, solver="lbfgs", max_iter=200)
    estimator.fit(logit, y)
    return {
        "slope": float(estimator.coef_[0, 0]),
        "intercept": float(estimator.intercept_[0]),
    }


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return []
    indexes = np.digitize(p, edges[1:-1], right=False)
    rows = []
    for bin_index in range(edges.size - 1):
        mask = indexes == bin_index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": int(bin_index),
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "fraction_positive": float(y[mask].mean()),
            }
        )
    return rows


def utility_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    threshold: float,
) -> float:
    predicted = (np.asarray(probabilities) >= threshold).astype(float)
    label_sequences, prediction_sequences = _sequences(labels, predicted, offsets)
    return float(normalized_utility(label_sequences, prediction_sequences))


def threshold_maximising_utility(
    labels: np.ndarray,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    grid: np.ndarray | None = None,
) -> dict:
    if grid is None:
        grid = np.linspace(0.01, 0.99, 50)
    best_threshold = float(grid[0])
    best_utility = -np.inf
    for threshold in grid:
        utility = utility_at_threshold(labels, probabilities, offsets, float(threshold))
        if utility > best_utility:
            best_utility = utility
            best_threshold = float(threshold)
    return {"threshold": best_threshold, "normalized_utility": float(best_utility)}


def warning_lead_hours(
    labels: np.ndarray,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    threshold: float,
) -> dict:
    """Hours from the first alarm to the reconstructed onset.

    Onset is the challenge reconstruction ``first positive hour + 6``.
    A positive lead means the first alarm is before that onset. Patients whose
    reconstructed onset falls outside the file are counted separately and do
    not enter the median.
    """
    predicted = np.asarray(probabilities) >= threshold
    leads: list[float] = []
    sepsis_patients = 0
    onset_inside = 0
    alarmed = 0
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        y = np.asarray(labels[start:stop])
        if not np.any(y == 1):
            continue
        sepsis_patients += 1
        onset = reconstructed_onset_index(y)
        if onset is None or onset < 0 or onset >= y.size:
            continue
        onset_inside += 1
        alarms = np.flatnonzero(predicted[start:stop])
        if alarms.size == 0:
            continue
        alarmed += 1
        leads.append(float(onset - int(alarms[0])))
    return {
        "n_sepsis_patients": sepsis_patients,
        "n_onset_inside_file": onset_inside,
        "n_with_alarm": alarmed,
        "median_lead_hours": float(np.median(leads)) if leads else None,
        "definition": "reconstructed_onset_index minus index of the first probability at or above the threshold",
    }


def _sequences(labels, predictions, offsets):
    label_sequences = []
    prediction_sequences = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        label_sequences.append(np.asarray(labels[start:stop], dtype=float))
        prediction_sequences.append(np.asarray(predictions[start:stop], dtype=float))
    return label_sequences, prediction_sequences
