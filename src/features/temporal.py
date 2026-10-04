"""Backward-looking temporal features.

Every function uses only indices <= t. Missing measurements stay missing.
They are not treated as zeros, and they are not filled from the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_float_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("temporal features expect a 1-D series")
    return array


def _check_time(t: int, n: int) -> int:
    if not isinstance(t, (int, np.integer)):
        raise TypeError("t must be an integer index")
    t = int(t)
    if t < 0 or t >= n:
        raise IndexError(f"t={t} is outside series length {n}")
    return t


@dataclass(frozen=True)
class ObservationState:
    """Representation of one variable at hour t without pretending missing hours were measured."""

    raw: float
    missing: int
    last_value: float
    time_since_measurement: float
    previous_distinct_value: float
    change_from_previous_measurement: float
    change_direction: float
    measurements_in_window: int


def observation_state(values: np.ndarray, t: int, recent_window: int = 24) -> ObservationState:
    """State at index t using observations at or before t only.

    raw is the measurement recorded at t, or NaN if that hour has no measurement.
    last_value is the most recent real measurement at or before t.
    time_since_measurement is 0 when t itself is measured, and NaN when no
    measurement exists yet. It is never coded as 0 just because the value is missing.
    change_* compares the last two real measurements when both exist.
    """
    series = _as_float_array(values)
    t = _check_time(t, series.size)
    if recent_window < 1:
        raise ValueError("recent_window must be positive")
    history = series[: t + 1]
    observed = np.flatnonzero(~np.isnan(history))
    raw = float(history[t])
    missing = int(np.isnan(raw))
    window_start = max(0, t - recent_window + 1)
    in_window = int(np.sum(observed >= window_start))
    if observed.size == 0:
        return ObservationState(
            raw=raw,
            missing=missing,
            last_value=np.nan,
            time_since_measurement=np.nan,
            previous_distinct_value=np.nan,
            change_from_previous_measurement=np.nan,
            change_direction=np.nan,
            measurements_in_window=0,
        )
    last_index = int(observed[-1])
    last_value = float(history[last_index])
    time_since = float(t - last_index)
    if observed.size >= 2:
        previous_index = int(observed[-2])
        previous_value = float(history[previous_index])
        change = last_value - previous_value
        direction = float(np.sign(change))
    else:
        previous_value = np.nan
        change = np.nan
        direction = np.nan
    return ObservationState(
        raw=raw,
        missing=missing,
        last_value=last_value,
        time_since_measurement=time_since,
        previous_distinct_value=previous_value,
        change_from_previous_measurement=change,
        change_direction=direction,
        measurements_in_window=in_window,
    )


def backward_observed(values: np.ndarray, t: int, window: int) -> np.ndarray:
    """Real measurements in the inclusive window ending at t. NaNs are dropped."""
    series = _as_float_array(values)
    t = _check_time(t, series.size)
    if window < 1:
        raise ValueError("window must be positive")
    start = max(0, t - window + 1)
    segment = series[start : t + 1]
    return segment[~np.isnan(segment)]


def backward_indices(t: int, window: int) -> tuple[int, int]:
    """Inclusive [start, t] index range. start is never greater than t and never negative."""
    if window < 1:
        raise ValueError("window must be positive")
    return max(0, t - window + 1), t


def backward_mean(values: np.ndarray, t: int, window: int) -> float:
    observed = backward_observed(values, t, window)
    if observed.size == 0:
        return np.nan
    return float(np.mean(observed))


def backward_median(values: np.ndarray, t: int, window: int) -> float:
    observed = backward_observed(values, t, window)
    if observed.size == 0:
        return np.nan
    return float(np.median(observed))


def backward_min(values: np.ndarray, t: int, window: int) -> float:
    observed = backward_observed(values, t, window)
    if observed.size == 0:
        return np.nan
    return float(np.min(observed))


def backward_max(values: np.ndarray, t: int, window: int) -> float:
    observed = backward_observed(values, t, window)
    if observed.size == 0:
        return np.nan
    return float(np.max(observed))


def backward_std(values: np.ndarray, t: int, window: int) -> float:
    observed = backward_observed(values, t, window)
    if observed.size < 2:
        return np.nan
    return float(np.std(observed, ddof=1))


def hour_delta(values: np.ndarray, t: int, lag: int) -> float:
    """Difference raw[t] - raw[t-lag] only when both hours were actually measured.

    A missing neighbor produces NaN. It does not fall back to an older carried value
    and it does not treat the missing hour as zero.
    """
    series = _as_float_array(values)
    t = _check_time(t, series.size)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    source = t - lag
    if source < 0 or np.isnan(series[t]) or np.isnan(series[source]):
        return np.nan
    return float(series[t] - series[source])


def backward_slope(values: np.ndarray, t: int, window: int) -> float:
    """Ordinary least-squares slope of observed points versus hour index, window ending at t.

    Requires at least two real measurements in the window. Future hours are not included.
    """
    series = _as_float_array(values)
    t = _check_time(t, series.size)
    if window < 2:
        raise ValueError("slope window must be at least 2")
    start = max(0, t - window + 1)
    segment = series[start : t + 1]
    indexes = np.flatnonzero(~np.isnan(segment))
    if indexes.size < 2:
        return np.nan
    x = indexes.astype(np.float64)
    y = segment[indexes]
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0:
        return np.nan
    return float(np.dot(x_centered, y - y.mean()) / denominator)


def candidate_feature_names(empirical_class: str, variable: str) -> list[str]:
    """Candidate features to test later. Usefulness is not established in phase 1."""
    if empirical_class == "static":
        return [f"{variable}_value", f"{variable}_missing"]
    if empirical_class == "time_context":
        return [variable]
    if empirical_class == "target":
        return []
    if empirical_class == "near_hourly":
        return [
            f"{variable}_raw",
            f"{variable}_missing",
            f"{variable}_last",
            f"{variable}_time_since_measurement",
            f"{variable}_delta_1h",
            f"{variable}_delta_3h",
            f"{variable}_mean_3h",
            f"{variable}_mean_6h",
            f"{variable}_median_6h",
            f"{variable}_min_6h",
            f"{variable}_max_6h",
            f"{variable}_std_6h",
            f"{variable}_slope_6h",
        ]
    if empirical_class == "intermittent":
        return [
            f"{variable}_raw",
            f"{variable}_missing",
            f"{variable}_last",
            f"{variable}_time_since_measurement",
            f"{variable}_change_from_previous_measurement",
            f"{variable}_change_direction",
            f"{variable}_measurements_in_24h",
            f"{variable}_mean_6h",
        ]
    if empirical_class == "sparse":
        return [
            f"{variable}_raw",
            f"{variable}_missing",
            f"{variable}_last",
            f"{variable}_time_since_measurement",
            f"{variable}_change_from_previous_measurement",
            f"{variable}_change_direction",
            f"{variable}_measurements_in_24h",
        ]
    raise ValueError(f"unknown empirical class {empirical_class}")
