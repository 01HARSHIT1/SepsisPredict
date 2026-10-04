"""Deterministic parameter masking for one flexible-input model.

A variable is removed by blanking every feature derived from it. Its missing
indicator is set to 1 and its measurement count is set to 0. A measured numeric
zero is left unchanged when the variable is kept. Missing is never written as a
measured zero.

Training uses seed 2019 and a fixed mixture, per row:

- 40% natural PhysioNet missingness only
- 20% availability-weighted dropout, with each variable blanked at its
  training-set missing rate from the Phase 1 observation counts
- 15% independent dropout at probability 0.5, so the expected number of
  retained variables is 20; this is one arm, not the whole training policy
- 25% a structured clinical-group or availability-count mask

Evaluation protocols that are not the training mixture use seed 2024.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.schema import LABS, STATIC, TIME_CONTEXT, VITALS
from src.inference.features import INPUT_COLUMNS
from src.training.masks import SELECTED_LABS

TRAIN_SEED = 2019
EVAL_SEED = 2024
NATURAL_FRACTION = 0.40
WEIGHTED_FRACTION = 0.20
UNIFORM_FRACTION = 0.15
UNIFORM_DROP_PROBABILITY = 0.5

RAPID_VITALS = ("HR", "O2Sat", "SBP", "MAP", "DBP", "Resp")
SPARSE_LABS = ("Phosphate", "Bilirubin_total")
MAJOR_ONE_VARIABLE = (
    "HR",
    "Temp",
    "MAP",
    "Resp",
    "O2Sat",
    "SBP",
    "DBP",
    "Lactate",
    "Creatinine",
    "WBC",
    "Platelets",
    "Bilirubin_total",
    "Glucose",
    "Age",
)
STRUCTURED_SCENARIOS = (
    "drop_vitals",
    "drop_labs",
    "drop_selected_labs",
    "only_rapid_vitals",
    "only_sparse_labs",
    "static_and_time_only",
    "keep_1_availability_ranked",
    "keep_2_availability_ranked",
    "keep_3_availability_ranked",
    "keep_5_availability_ranked",
    "keep_10_availability_ranked",
    "keep_15_availability_ranked",
    "keep_20_availability_ranked",
    "keep_30_availability_ranked",
    "one_major_variable",
)


@dataclass(frozen=True)
class FeatureLayout:
    names: list[str]
    variables: list[str]
    columns_by_variable: dict[str, list[int]]
    missing_column: dict[str, int]
    count_column: dict[str, int]


def layout_from_spec(spec: dict) -> FeatureLayout:
    names = [item["name"] for item in spec["features"]]
    columns: dict[str, list[int]] = {}
    missing: dict[str, int] = {}
    counts: dict[str, int] = {}
    for index, item in enumerate(spec["features"]):
        columns.setdefault(item["variable"], []).append(index)
        if item["name"].endswith("_missing"):
            missing[item["variable"]] = index
        if item["name"].endswith("measurements_in_24h"):
            counts[item["variable"]] = index
    missing_variables = [name for name in INPUT_COLUMNS if name not in columns]
    if missing_variables:
        raise ValueError(f"feature spec has no columns for {missing_variables}")
    return FeatureLayout(names, list(INPUT_COLUMNS), columns, missing, counts)


def protocol_description(observation_rates: list[float]) -> dict:
    return {
        "train_seed": TRAIN_SEED,
        "evaluation_seed": EVAL_SEED,
        "natural_fraction": NATURAL_FRACTION,
        "availability_weighted_fraction": WEIGHTED_FRACTION,
        "uniform_fraction": UNIFORM_FRACTION,
        "uniform_drop_probability": UNIFORM_DROP_PROBABILITY,
        "uniform_justification": (
            "Probability 0.5 retains 20 variables in expectation, the middle of the "
            "requested information curve. Uniform dropout is 15% of training rows."
        ),
        "availability_weighted_rule": (
            "Blank variable i with probability 1 - training observation rate. "
            "Rates come from the Phase 1 hourly observation counts."
        ),
        "observation_rates": {
            name: float(rate) for name, rate in zip(INPUT_COLUMNS, observation_rates, strict=True)
        },
        "structured_scenarios": list(STRUCTURED_SCENARIOS),
        "major_one_variable": list(MAJOR_ONE_VARIABLE),
    }


def training_drop_mask(
    row_indices: np.ndarray,
    observation_rates: np.ndarray,
    availability_ranked: list[str],
    seed: int = TRAIN_SEED,
) -> np.ndarray:
    """Return a boolean matrix, True where that variable is artificially removed."""
    rows = np.asarray(row_indices, dtype=np.int64)
    rates = np.asarray(observation_rates, dtype=np.float64)
    if rates.shape != (len(INPUT_COLUMNS),):
        raise ValueError("observation rates must cover the 40 inputs")
    draws = _uniform_matrix(rows, 4 + len(INPUT_COLUMNS), seed)
    policy = draws[:, 0]
    drop = np.zeros((rows.shape[0], len(INPUT_COLUMNS)), dtype=bool)
    weighted = (policy >= NATURAL_FRACTION) & (policy < NATURAL_FRACTION + WEIGHTED_FRACTION)
    uniform_end = NATURAL_FRACTION + WEIGHTED_FRACTION + UNIFORM_FRACTION
    uniform_arm = (policy >= NATURAL_FRACTION + WEIGHTED_FRACTION) & (policy < uniform_end)
    structured = policy >= uniform_end
    variable_draw = draws[:, 4:]
    drop[weighted] = variable_draw[weighted] < (1.0 - rates)
    drop[uniform_arm] = variable_draw[uniform_arm] < UNIFORM_DROP_PROBABILITY
    if np.any(structured):
        scenario_draw = draws[:, 1]
        choice_draw = draws[:, 2]
        for row in np.flatnonzero(structured):
            scenario_index = min(
                int(scenario_draw[row] * len(STRUCTURED_SCENARIOS)),
                len(STRUCTURED_SCENARIOS) - 1,
            )
            drop[row] = _scenario_drop(
                STRUCTURED_SCENARIOS[scenario_index],
                float(choice_draw[row]),
                availability_ranked,
            )
    return drop


def evaluation_independent_drop(
    row_indices: np.ndarray,
    probabilities: np.ndarray,
    seed: int,
) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return _uniform_matrix(rows, len(INPUT_COLUMNS), seed) < probabilities


def scenario_drop_rows(n_rows: int, scenario: str, availability_ranked: list[str]) -> np.ndarray:
    pattern = _scenario_drop(scenario, 0.0, availability_ranked)
    return np.repeat(pattern[None, :], n_rows, axis=0)


def keep_variables_drop(n_rows: int, keep: set[str]) -> np.ndarray:
    unknown = keep.difference(INPUT_COLUMNS)
    if unknown:
        raise ValueError(f"unsupported variables: {sorted(unknown)}")
    drop = np.ones((n_rows, len(INPUT_COLUMNS)), dtype=bool)
    for index, name in enumerate(INPUT_COLUMNS):
        if name in keep:
            drop[:, index] = False
    return drop


def apply_drop_mask(block: np.ndarray, drop: np.ndarray, layout: FeatureLayout) -> np.ndarray:
    """Blank dropped variables in a copy. Kept measured zeros stay zero."""
    if block.shape[0] != drop.shape[0]:
        raise ValueError("mask row count does not match the feature block")
    out = np.array(block, dtype=np.float32, copy=True)
    for index, variable in enumerate(layout.variables):
        rows = np.flatnonzero(drop[:, index])
        if rows.size == 0:
            continue
        columns = layout.columns_by_variable[variable]
        out[np.ix_(rows, columns)] = np.nan
        missing_column = layout.missing_column.get(variable)
        if missing_column is not None:
            out[rows, missing_column] = 1.0
        count_column = layout.count_column.get(variable)
        if count_column is not None:
            out[rows, count_column] = 0.0
    return out


def _scenario_drop(scenario: str, choice: float, availability_ranked: list[str]) -> np.ndarray:
    if scenario == "drop_vitals":
        return _drop_these(set(VITALS))
    if scenario == "drop_labs":
        return _drop_these(set(LABS))
    if scenario == "drop_selected_labs":
        return _drop_these(set(SELECTED_LABS))
    if scenario == "only_rapid_vitals":
        return _keep_these(set(RAPID_VITALS))
    if scenario == "only_sparse_labs":
        return _keep_these(set(SPARSE_LABS))
    if scenario == "static_and_time_only":
        return _keep_these(set(STATIC) | set(TIME_CONTEXT))
    if scenario.startswith("keep_") and scenario.endswith("_availability_ranked"):
        count = int(scenario.split("_")[1])
        return _keep_these(set(availability_ranked[:count]))
    if scenario == "one_major_variable":
        choice_index = min(int(choice * len(MAJOR_ONE_VARIABLE)), len(MAJOR_ONE_VARIABLE) - 1)
        return _keep_these({MAJOR_ONE_VARIABLE[choice_index]})
    raise ValueError(f"unknown structured scenario {scenario}")


def _drop_these(dropped: set[str]) -> np.ndarray:
    return np.asarray([name in dropped for name in INPUT_COLUMNS], dtype=bool)


def _keep_these(kept: set[str]) -> np.ndarray:
    return np.asarray([name not in kept for name in INPUT_COLUMNS], dtype=bool)


def _uniform_matrix(row_indices: np.ndarray, n_columns: int, seed: int) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.uint64)[:, None]
    columns = np.arange(n_columns, dtype=np.uint64)[None, :]
    mixed = rows * np.uint64(0x9E3779B97F4A7C15) + columns * np.uint64(0xBF58476D1CE4E5B9)
    mixed = mixed ^ np.uint64(seed)
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    mixed = mixed ^ (mixed >> np.uint64(31))
    return (mixed >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
