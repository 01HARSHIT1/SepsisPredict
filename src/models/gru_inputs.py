"""Network matrix for the GRU.

Non-finite measurement values are scaled, then filled with 0. Missing
indicators stay 0 or 1. A filled 0 is therefore not a measured zero: the
missing indicator distinguishes them. Sparse labs keep last value and time
since measurement as separate columns. This is not hourly forward-fill of a
new observation.
"""

from __future__ import annotations

import numpy as np

from src.data.schema import LABS, STATIC, VITALS


def gru_feature_names(spec: dict) -> list[str]:
    present = {item["name"] for item in spec["features"]}
    names: list[str] = []
    for variable in VITALS + LABS:
        for suffix in ("raw", "missing", "last", "time_since_measurement"):
            name = f"{variable}_{suffix}"
            if name in present:
                names.append(name)
    for variable in STATIC:
        for suffix in ("value", "missing"):
            name = f"{variable}_{suffix}"
            if name in present:
                names.append(name)
    if "ICULOS" in present:
        names.append("ICULOS")
    if not names:
        raise ValueError("feature spec does not contain GRU inputs")
    return names


def fit_column_scaler(values: np.ndarray, names: list[str]) -> dict[str, tuple[float, float]]:
    scaler: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(names):
        if name.endswith("_missing"):
            scaler[name] = (0.0, 1.0)
            continue
        column = np.asarray(values[:, index], dtype=np.float64)
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            scaler[name] = (0.0, 1.0)
            continue
        mean = float(finite.mean())
        scale = float(finite.std())
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        scaler[name] = (mean, scale)
    return scaler


def to_network_inputs(values: np.ndarray, names: list[str], scaler: dict[str, tuple[float, float]]) -> np.ndarray:
    out = np.zeros((values.shape[0], len(names)), dtype=np.float32)
    for index, name in enumerate(names):
        column = np.asarray(values[:, index], dtype=np.float64)
        if name.endswith("_missing"):
            filled = np.where(np.isfinite(column), column, 1.0)
            out[:, index] = filled
            continue
        mean, scale = scaler[name]
        finite = np.isfinite(column)
        scaled = np.zeros(column.shape[0], dtype=np.float64)
        scaled[finite] = (column[finite] - mean) / scale
        out[:, index] = scaled
    return out


def missing_pad_row(names: list[str], scaler: dict[str, tuple[float, float]]) -> np.ndarray:
    blank = np.full((1, len(names)), np.nan, dtype=np.float64)
    return to_network_inputs(blank, names, scaler)[0]
