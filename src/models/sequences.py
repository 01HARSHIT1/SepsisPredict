"""Chronological windows for one patient.

Index t uses hours at or before t. The left side is padded when the stay is
shorter than the window. Padding does not drop those hours, and it does not
read a later hour.
"""

from __future__ import annotations

import numpy as np


def window_coverage(lengths: np.ndarray, window: int) -> dict:
    lengths = np.asarray(lengths, dtype=np.int64)
    if window < 1:
        raise ValueError("window must be positive")
    scored = int(lengths.sum())
    full = int(np.maximum(lengths - window + 1, 0).sum())
    padded = scored - full
    return {
        "window": int(window),
        "patients": int(lengths.size),
        "scored_hours": scored,
        "hours_with_full_window": full,
        "hours_with_left_padding": padded,
        "fraction_hours_with_left_padding": padded / scored if scored else None,
        "patients_shorter_than_window": int(np.sum(lengths < window)),
    }


def windows_ending_at_each_hour(
    values: np.ndarray,
    window: int,
    pad_row: np.ndarray,
) -> np.ndarray:
    """Return shape (hours, window, features). Row t ends at hour t."""
    if values.ndim != 2:
        raise ValueError("values must be (hours, features)")
    if window < 1:
        raise ValueError("window must be positive")
    hours, features = values.shape
    if pad_row.shape != (features,):
        raise ValueError("pad_row must match the feature width")
    padded = np.repeat(np.asarray(pad_row, dtype=np.float32)[None, :], hours + window - 1, axis=0)
    padded[window - 1 :] = np.asarray(values, dtype=np.float32)
    view = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return np.moveaxis(view, -1, 1)


def patient_block(features: np.ndarray, offsets: np.ndarray, patient_index: int) -> np.ndarray:
    start = int(offsets[patient_index])
    stop = int(offsets[patient_index + 1])
    return np.array(features[start:stop], copy=True)
