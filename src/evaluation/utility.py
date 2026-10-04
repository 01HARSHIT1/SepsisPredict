"""PhysioNet/CinC 2019 utility score.

The numerical definition follows the public challenge scorer
(physionetchallenges/evaluation-2019, evaluate_sepsis_score.py) and the
description on the PhysioNet challenge page. A normalized score of 1 is the
best in-range policy and 0 is no positive predictions. This module does not
score a trained model in phase 1; tests check the published example.
"""

from __future__ import annotations

import numpy as np

from src.labels.targets import DT_EARLY, DT_LATE, DT_OPTIMAL

MAX_U_TP = 1.0
MIN_U_FN = -2.0
U_FP = -0.05
U_TN = 0.0


def compute_prediction_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: float = DT_EARLY,
    dt_optimal: float = DT_OPTIMAL,
    dt_late: float = DT_LATE,
    max_u_tp: float = MAX_U_TP,
    min_u_fn: float = MIN_U_FN,
    u_fp: float = U_FP,
    u_tn: float = U_TN,
) -> float:
    """Total utility of one patient's hourly binary predictions."""
    y = np.asarray(labels, dtype=float)
    yhat = np.asarray(predictions, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("labels and predictions must have the same length")
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be 0 or 1")
    if not np.all(np.isin(yhat, (0, 1))):
        raise ValueError("predictions must be binary 0 or 1")
    if dt_early >= dt_optimal or dt_optimal >= dt_late:
        raise ValueError("utility time cutpoints must satisfy early < optimal < late")

    if np.any(y == 1):
        is_septic = True
        t_sepsis = float(int(np.argmax(y == 1)) - dt_optimal)
    else:
        is_septic = False
        t_sepsis = float("inf")

    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    utility = np.zeros(y.size, dtype=float)
    for t in range(y.size):
        if t > t_sepsis + dt_late:
            continue
        positive = yhat[t] == 1
        if is_septic and positive:
            if t <= t_sepsis + dt_optimal:
                utility[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
            else:
                utility[t] = m_2 * (t - t_sepsis) + b_2
        elif (not is_septic) and positive:
            utility[t] = u_fp
        elif is_septic and not positive:
            if t <= t_sepsis + dt_optimal:
                utility[t] = 0.0
            else:
                utility[t] = m_3 * (t - t_sepsis) + b_3
        else:
            utility[t] = u_tn
    return float(np.sum(utility))


def best_predictions(labels: np.ndarray) -> np.ndarray:
    """In-range positive calls used as the utility normalizer for one patient."""
    y = np.asarray(labels, dtype=float)
    best = np.zeros(y.size, dtype=float)
    if not np.any(y == 1):
        return best
    t_sepsis = int(np.argmax(y == 1)) - DT_OPTIMAL
    start = max(0, t_sepsis + DT_EARLY)
    stop = min(y.size, t_sepsis + DT_LATE + 1)
    if stop > start:
        best[start:stop] = 1
    return best


def assert_published_utility_example() -> float:
    """The challenge scorer's documented example must reproduce exactly."""
    labels = np.array([0, 0, 0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1, 1, 1])
    value = compute_prediction_utility(labels, predictions)
    expected = 3.388888888888889
    if abs(value - expected) > 1e-12:
        raise AssertionError(f"utility example returned {value}, expected {expected}")
    return value


def normalized_utility(label_sequences: list[np.ndarray], prediction_sequences: list[np.ndarray]) -> float:
    """Cohort utility normalized so the best in-range policy scores 1 and inaction scores 0."""
    if len(label_sequences) != len(prediction_sequences):
        raise ValueError("each patient needs labels and predictions")
    observed = 0.0
    best = 0.0
    inaction = 0.0
    for labels, predictions in zip(label_sequences, prediction_sequences, strict=True):
        observed += compute_prediction_utility(labels, predictions)
        best += compute_prediction_utility(labels, best_predictions(labels))
        inaction += compute_prediction_utility(labels, np.zeros(len(labels)))
    denominator = best - inaction
    if denominator == 0:
        return float("nan")
    return (observed - inaction) / denominator
