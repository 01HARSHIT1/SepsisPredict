"""Label definitions for the PhysioNet/CinC 2019 early-prediction task.

The official SepsisLabel is not a marker of confirmed sepsis at the current hour.
For patients who meet the challenge sepsis definition, SepsisLabel becomes 1 at
t_sepsis - 6 hours and stays 1 afterward. For everyone else it is 0.

t_sepsis itself is not a column. The challenge utility code reconstructs it as
the index of the first positive label plus 6 hours. That reconstruction is an
evaluation target, not a model input.
"""

from __future__ import annotations

import numpy as np

# Official utility parameters from the 2019 challenge scoring code.
DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3


def first_positive_index(labels: np.ndarray) -> int | None:
    """0-based index of the first SepsisLabel == 1, or None if the stay is never positive."""
    array = np.asarray(labels)
    matches = np.flatnonzero(array == 1)
    if matches.size == 0:
        return None
    return int(matches[0])


def reconstructed_onset_index(labels: np.ndarray) -> int | None:
    """0-based challenge onset index: first positive label index - dt_optimal.

    dt_optimal is -6, so this is first_positive + 6. The result may be past the
    end of the record when the file does not contain the reconstructed onset.
    """
    first = first_positive_index(labels)
    if first is None:
        return None
    return int(first - DT_OPTIMAL)


def label_is_monotonic_nondecreasing(labels: np.ndarray) -> bool:
    array = np.asarray(labels, dtype=float)
    if array.size <= 1:
        return True
    return bool(np.all(np.diff(array) >= 0))


def future_positive_within_horizon(labels: np.ndarray, t: int, horizon_hours: int) -> int:
    """Target: 1 if any official positive label occurs in hours (t, t+horizon].

    This function reads future labels on purpose because it defines a prediction
    target. It must not be passed into a feature builder.
    Hours at or before t are not counted, so the current label is not copied forward
    as if it were a new future event.
    """
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be positive")
    array = np.asarray(labels)
    if t < 0 or t >= array.size:
        raise IndexError("t outside labels")
    end = min(array.size, t + horizon_hours + 1)
    if end <= t + 1:
        return 0
    return int(np.any(array[t + 1 : end] == 1))
