"""Vectorized feature rows for one patient.

The values match ``src.inference.features._feature_value`` at every hour.
Windows end at the current hour. Missing entries stay missing.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from src.inference.features import INPUT_INDEX


def assemble_patient(measurements: np.ndarray, spec: dict) -> np.ndarray:
    """Return a float64 array of shape (hours, n_features) in spec order."""
    if measurements.ndim != 2:
        raise ValueError("measurements must be a 2-D hourly array")
    features = spec["features"]
    n_hours = measurements.shape[0]
    out = np.full((n_hours, len(features)), np.nan, dtype=np.float64)
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for index, item in enumerate(features):
        grouped.setdefault(item["variable"], []).append((index, item))
    for variable, items in grouped.items():
        needed = {_suffix(item) for _, item in items}
        derived = derive_series(measurements[:, INPUT_INDEX[variable]], needed)
        for index, item in items:
            out[:, index] = _column(derived, item)
    return out


def _suffix(item: dict) -> str:
    if item["empirical_class"] == "time_context":
        return "raw"
    if item["empirical_class"] == "static":
        return "static_value" if item["name"].endswith("_value") else "static_missing"
    return item["name"][len(item["variable"]) + 1 :]


def derive_series(series: np.ndarray, needed: set[str] | None = None) -> dict[str, np.ndarray]:
    values = np.asarray(series, dtype=np.float64)
    n_hours = values.shape[0]
    observed = ~np.isnan(values)
    position = np.arange(n_hours)
    last_index = np.full(n_hours, -1, dtype=np.int32)
    previous_index = np.full(n_hours, -1, dtype=np.int32)
    running_last = -1
    running_previous = -1
    for hour in range(n_hours):
        if observed[hour]:
            running_previous = running_last
            running_last = hour
        last_index[hour] = running_last
        previous_index[hour] = running_previous

    last = np.full(n_hours, np.nan, dtype=np.float64)
    valid_last = last_index >= 0
    last[valid_last] = values[last_index[valid_last]]
    time_since = np.full(n_hours, np.nan, dtype=np.float64)
    time_since[valid_last] = position[valid_last] - last_index[valid_last]

    change = np.full(n_hours, np.nan, dtype=np.float64)
    valid_previous = previous_index >= 0
    change[valid_previous] = last[valid_previous] - values[previous_index[valid_previous]]
    direction = np.full(n_hours, np.nan, dtype=np.float64)
    direction[valid_previous] = np.sign(change[valid_previous])

    cumulative = np.cumsum(observed.astype(np.int32))
    count_24 = cumulative.astype(np.float64)
    if n_hours > 24:
        count_24[24:] = cumulative[24:] - cumulative[:-24]

    window_3 = _rolling(values, 3) if _needs_window(needed, 3) else None
    window_6 = _rolling(values, 6, mean_only=_mean_only(needed)) if _needs_window(needed, 6) else None
    empty = np.full(n_hours, np.nan, dtype=np.float64)
    return {
        "raw": values,
        "missing": (~observed).astype(np.float64),
        "last": last,
        "time_since_measurement": time_since,
        "change_from_previous_measurement": change,
        "change_direction": direction,
        "measurements_in_24h": count_24,
        "static_value": last,
        "static_missing": (~valid_last).astype(np.float64),
        "delta_1h": _lag_delta(values, observed, 1),
        "delta_3h": _lag_delta(values, observed, 3),
        "mean_3h": empty if window_3 is None else window_3["mean"],
        "mean_6h": empty if window_6 is None else window_6["mean"],
        "median_6h": empty if window_6 is None else window_6["median"],
        "min_6h": empty if window_6 is None else window_6["min"],
        "max_6h": empty if window_6 is None else window_6["max"],
        "std_6h": empty if window_6 is None else window_6["std"],
        "slope_6h": empty if window_6 is None else window_6["slope"],
    }


def _needs_window(needed: set[str] | None, window: int) -> bool:
    if needed is None:
        return True
    if window == 3:
        return "mean_3h" in needed
    return bool(needed & {"mean_6h", "median_6h", "min_6h", "max_6h", "std_6h", "slope_6h"})


def _mean_only(needed: set[str] | None) -> bool:
    if needed is None:
        return False
    return "mean_6h" in needed and not (needed & {"median_6h", "min_6h", "max_6h", "std_6h", "slope_6h"})


def _lag_delta(values: np.ndarray, observed: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.shape[0] <= lag:
        return out
    both = observed[lag:] & observed[:-lag]
    delta = values[lag:] - values[:-lag]
    out[lag:] = np.where(both, delta, np.nan)
    return out


def _rolling(values: np.ndarray, window: int, mean_only: bool = False) -> dict[str, np.ndarray]:
    mean, median, minimum, maximum, std, slope = _rolling_numba(
        np.ascontiguousarray(values, dtype=np.float64), window, mean_only
    )
    return {"mean": mean, "median": median, "min": minimum, "max": maximum, "std": std, "slope": slope}


@njit(cache=True)
def _rolling_numba(values, window, mean_only):
    n_hours = values.shape[0]
    mean = np.empty(n_hours, dtype=np.float64)
    median = np.empty(n_hours, dtype=np.float64)
    minimum = np.empty(n_hours, dtype=np.float64)
    maximum = np.empty(n_hours, dtype=np.float64)
    std = np.empty(n_hours, dtype=np.float64)
    slope = np.empty(n_hours, dtype=np.float64)
    mean[:] = np.nan
    median[:] = np.nan
    minimum[:] = np.nan
    maximum[:] = np.nan
    std[:] = np.nan
    slope[:] = np.nan
    buffer = np.empty(window, dtype=np.float64)
    positions = np.empty(window, dtype=np.float64)
    for hour in range(n_hours):
        start = hour - window + 1
        if start < 0:
            start = 0
        count = 0
        for index in range(start, hour + 1):
            value = values[index]
            if not np.isnan(value):
                buffer[count] = value
                positions[count] = index - start
                count += 1
        if count == 0:
            continue
        total = 0.0
        low = buffer[0]
        high = buffer[0]
        for index in range(count):
            value = buffer[index]
            total += value
            if value < low:
                low = value
            if value > high:
                high = value
        center = total / count
        mean[hour] = center
        if mean_only:
            continue
        minimum[hour] = low
        maximum[hour] = high
        if count >= 2:
            residual = 0.0
            x_total = 0.0
            for index in range(count):
                residual += (buffer[index] - center) * (buffer[index] - center)
                x_total += positions[index]
            std[hour] = np.sqrt(residual / (count - 1))
            x_center = x_total / count
            denominator = 0.0
            numerator = 0.0
            for index in range(count):
                centered = positions[index] - x_center
                denominator += centered * centered
                numerator += centered * (buffer[index] - center)
            if denominator != 0.0:
                slope[hour] = numerator / denominator
        median[hour] = _median_sorted(buffer, count)
    return mean, median, minimum, maximum, std, slope


@njit(cache=True)
def _median_sorted(buffer, count):
    for index in range(1, count):
        value = buffer[index]
        position = index
        while position > 0 and buffer[position - 1] > value:
            buffer[position] = buffer[position - 1]
            position -= 1
        buffer[position] = value
    if count % 2 == 1:
        return buffer[count // 2]
    right = count // 2
    return 0.5 * (buffer[right - 1] + buffer[right])


def _column(derived: dict[str, np.ndarray], item: dict) -> np.ndarray:
    empirical_class = item["empirical_class"]
    name = item["name"]
    variable = item["variable"]
    if empirical_class == "time_context":
        return derived["raw"]
    if empirical_class == "static":
        if name.endswith("_value"):
            return derived["static_value"]
        if name.endswith("_missing"):
            return derived["static_missing"]
        raise ValueError(f"unsupported static feature {name}")
    suffix = name[len(variable) + 1 :]
    if suffix not in derived:
        raise ValueError(f"unsupported feature {name}")
    return derived[suffix]
