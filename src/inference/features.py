"""Turn hourly measurements into the feature vector training and inference share.

Missing measurements stay missing. A numeric zero is a real zero.
Omitted keys and JSON null are missing. Future hours are not readable.
"""

from __future__ import annotations

import math

import numpy as np

from src.data.schema import ALL_SPECS, TARGET
from src.features.temporal import (
    backward_max,
    backward_mean,
    backward_median,
    backward_min,
    backward_slope,
    backward_std,
    hour_delta,
    observation_state,
)

INPUT_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in ALL_SPECS if spec.name != TARGET)
INPUT_INDEX: dict[str, int] = {name: index for index, name in enumerate(INPUT_COLUMNS)}
DATA_TOTAL = len(INPUT_COLUMNS)
MAX_TIMELINE_HOURS = 720


def parse_observations(payload: dict | list) -> np.ndarray:
    """Hourly input matrix. Gap hours are inserted as missing when hour keys are used."""
    measurements, _, _ = parse_request(payload)
    return measurements


def parse_request(payload: dict | list) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    """Return measurements, scored row indexes, and submitted hour labels.

    Accepted bodies:

    - ``{"HR": 118, "Temp": 38.7}`` current sparse values
    - ``{"observations": [{"values": {"HR": 118}}]}`` one or more hours in order
    - ``[{"hour": 1, "HR": 88}, {"hour": 3, "HR": 101}]`` explicit hours

    When any hour is present, every item needs a unique integer hour and the
    submitted order must already be chronological. Missing hours between the
    first and last become rows of missing values, so a one-hour change is not
    computed across a gap. Only submitted hours are scored. Omitted parameters
    stay missing. Numeric 0 is a measured zero.
    """
    records = _extract_records(payload)
    rows = [_values_row(record) for record in records]
    if any("hour" in record for record in records):
        if any("hour" not in record for record in records):
            raise ValueError("hour must be present on every observation or on none")
        hours = [_hour_number(record["hour"]) for record in records]
        if len(set(hours)) != len(hours):
            raise ValueError("hour values must be unique")
        if any(later <= earlier for earlier, later in zip(hours, hours[1:], strict=False)):
            raise ValueError("observations must be in chronological order")
        span = hours[-1] - hours[0] + 1
        if span > MAX_TIMELINE_HOURS:
            raise ValueError(f"timeline span must be at most {MAX_TIMELINE_HOURS} hours")
        measurements = np.full((span, DATA_TOTAL), np.nan, dtype=np.float64)
        scored: list[int] = []
        labels: list[int | None] = []
        for hour, row in zip(hours, rows, strict=True):
            index = hour - hours[0]
            measurements[index] = row
            scored.append(index)
            labels.append(hour)
        return measurements, np.asarray(scored, dtype=np.int64), labels
    if len(rows) > MAX_TIMELINE_HOURS:
        raise ValueError(f"timeline span must be at most {MAX_TIMELINE_HOURS} hours")
    measurements = np.vstack(rows)
    scored = np.arange(measurements.shape[0], dtype=np.int64)
    return measurements, scored, [None] * measurements.shape[0]


def _extract_records(payload: dict | list) -> list[dict]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and "observations" in payload:
        records = payload["observations"]
    elif isinstance(payload, dict) and "values" in payload:
        records = [payload]
    elif isinstance(payload, dict) and _is_parameter_map(payload):
        records = [payload]
    else:
        raise ValueError(
            "request must be parameter values, an observations list, or a list of hourly objects"
        )
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("observations must be a non-empty list in chronological order")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("each observation must be an object")
    return records


def _is_parameter_map(payload: dict) -> bool:
    allowed = set(INPUT_COLUMNS) | {"hour"}
    return len(payload) > 0 and set(payload).issubset(allowed)


def _hour_number(raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("hour must be an integer")
    if isinstance(raw, float) and (not math.isfinite(raw) or not raw.is_integer()):
        raise ValueError("hour must be an integer")
    hour = int(raw)
    if hour < 0:
        raise ValueError("hour must be zero or a positive integer")
    return hour


def _values_row(record: dict) -> np.ndarray:
    if "values" in record:
        if any(key in INPUT_INDEX for key in record):
            raise ValueError("put parameters inside values, or beside hour, not both")
        values = record["values"]
    else:
        values = {key: value for key, value in record.items() if key != "hour"}
    if not isinstance(values, dict):
        raise ValueError("values must be an object of parameter names to numbers or null")
    row = np.full(DATA_TOTAL, np.nan, dtype=np.float64)
    for name, raw in values.items():
        if name == TARGET:
            raise ValueError("SepsisLabel is a target and cannot be submitted as an input")
        if name not in INPUT_INDEX:
            raise ValueError(f"unsupported parameter: {name}")
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name} must be a number or null")
        if isinstance(raw, float) and (math.isnan(raw) or not math.isfinite(raw)):
            if math.isnan(raw):
                continue
            raise ValueError(f"{name} must be a finite number or null")
        row[INPUT_INDEX[name]] = float(raw)
    return row


def availability_at(measurements: np.ndarray, t: int) -> dict:
    """Count parameters present on hour t. Carried values are not recounted.

    ``data_sufficiency`` is the technical band from ``information_descriptor``.
    It is not clinical confidence. Completeness is the ratio of present values
    to the 40 supported inputs.
    """
    from src.inference.sufficiency import information_descriptor

    _check_time(measurements, t)
    return information_descriptor(measurements, t)


def feature_names(spec: dict) -> list[str]:
    return [item["name"] for item in spec["features"]]


def build_feature_row(measurements: np.ndarray, t: int, spec: dict) -> dict[str, float]:
    """Feature dict at hour t. Uses only rows 0..t of ``measurements``."""
    _check_time(measurements, t)
    if measurements.shape[1] != DATA_TOTAL:
        raise ValueError(f"expected {DATA_TOTAL} input columns")
    history = measurements[: t + 1]
    values: dict[str, float] = {}
    for item in spec["features"]:
        variable = item["variable"]
        column = history[:, INPUT_INDEX[variable]]
        # Pad so index t refers to the current hour inside the truncated history.
        values[item["name"]] = _feature_value(column, t, item)
    return values


def build_feature_matrix(measurements: np.ndarray, spec: dict) -> list[dict[str, float]]:
    """One feature row per hour, each row blind to later hours."""
    from src.features.assemble import assemble_patient

    array = assemble_patient(measurements, spec)
    names = feature_names(spec)
    rows: list[dict[str, float]] = []
    for hour in range(array.shape[0]):
        rows.append({name: float(array[hour, index]) for index, name in enumerate(names)})
    return rows


def _check_time(measurements: np.ndarray, t: int) -> None:
    if measurements.ndim != 2 or t < 0 or t >= measurements.shape[0]:
        raise IndexError("t is outside the supplied hourly record")


def _feature_value(series: np.ndarray, t: int, item: dict) -> float:
    empirical_class = item["empirical_class"]
    name = item["name"]
    variable = item["variable"]
    if empirical_class == "time_context":
        return float(series[t])
    state = observation_state(series, t, recent_window=24)
    if empirical_class == "static":
        if name.endswith("_value"):
            return float(state.last_value)
        if name.endswith("_missing"):
            return float(np.isnan(state.last_value))
        raise ValueError(f"unsupported static feature {name}")
    suffix = name[len(variable) + 1 :]
    computed = {
        "raw": state.raw,
        "missing": float(state.missing),
        "last": state.last_value,
        "time_since_measurement": state.time_since_measurement,
        "change_from_previous_measurement": state.change_from_previous_measurement,
        "change_direction": state.change_direction,
        "measurements_in_24h": float(state.measurements_in_window),
        "delta_1h": hour_delta(series, t, 1),
        "delta_3h": hour_delta(series, t, 3),
        "mean_3h": backward_mean(series, t, 3),
        "mean_6h": backward_mean(series, t, 6),
        "median_6h": backward_median(series, t, 6),
        "min_6h": backward_min(series, t, 6),
        "max_6h": backward_max(series, t, 6),
        "std_6h": backward_std(series, t, 6),
        "slope_6h": backward_slope(series, t, 6),
    }
    if suffix not in computed:
        raise ValueError(f"unsupported feature {name}")
    return float(computed[suffix])
