"""Patient-level splits.

Hourly rows from the same patient always stay in one partition.
The split is stratified by hospital system and by whether the patient has any
positive SepsisLabel, so those margins are preserved without putting one patient
in two partitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import SPLIT_FRACTIONS, SPLIT_SEED


@dataclass(frozen=True)
class SplitResult:
    assignments: pd.DataFrame
    summary: dict


def _allocate(n: int, fractions: tuple[float, float, float] = SPLIT_FRACTIONS) -> tuple[int, int, int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    raw = [n * fraction for fraction in fractions]
    counts = [int(np.floor(value)) for value in raw]
    remainder = n - sum(counts)
    fractional = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in fractional[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def assign_patient_splits(
    patients: pd.DataFrame,
    seed: int = SPLIT_SEED,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> SplitResult:
    """Assign each patient_id to train, validation, or test.

    patients must contain patient_id, hospital_system, and sepsis_patient.
    """
    required = {"patient_id", "hospital_system", "sepsis_patient"}
    missing = required.difference(patients.columns)
    if missing:
        raise ValueError(f"patients frame is missing {sorted(missing)}")
    if patients["patient_id"].duplicated().any():
        raise ValueError("duplicate patient_id values")

    rng = np.random.default_rng(seed)
    frame = patients.loc[:, ["patient_id", "hospital_system", "sepsis_patient"]].copy()
    frame["split"] = ""
    strata = frame.groupby(["hospital_system", "sepsis_patient"], sort=True).groups
    for key in sorted(strata, key=lambda item: (str(item[0]), int(item[1]))):
        positions = np.asarray(list(strata[key]), dtype=int)
        order = rng.permutation(positions.size)
        shuffled = positions[order]
        n_train, n_validation, n_test = _allocate(shuffled.size, fractions)
        if n_train + n_validation + n_test != shuffled.size:
            raise AssertionError("split sizes do not cover the stratum")
        train_ids = shuffled[:n_train]
        validation_ids = shuffled[n_train : n_train + n_validation]
        test_ids = shuffled[n_train + n_validation :]
        if test_ids.size != n_test:
            raise AssertionError("test allocation mismatch")
        frame.loc[train_ids, "split"] = "train"
        frame.loc[validation_ids, "split"] = "validation"
        frame.loc[test_ids, "split"] = "test"

    if (frame["split"] == "").any():
        raise AssertionError("some patients were not assigned")
    _assert_disjoint(frame)

    counts = frame["split"].value_counts().to_dict()
    summary = {
        "seed": seed,
        "fractions": {"train": fractions[0], "validation": fractions[1], "test": fractions[2]},
        "stratification": ["hospital_system", "sepsis_patient"],
        "n_patients": int(len(frame)),
        "n_train": int(counts.get("train", 0)),
        "n_validation": int(counts.get("validation", 0)),
        "n_test": int(counts.get("test", 0)),
        "overlap_patient_count": 0,
    }
    return SplitResult(assignments=frame.sort_values("patient_id").reset_index(drop=True), summary=summary)


def _assert_disjoint(frame: pd.DataFrame) -> None:
    grouped = {name: set(part["patient_id"]) for name, part in frame.groupby("split")}
    names = list(grouped)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = grouped[left] & grouped[right]
            if overlap:
                raise AssertionError(f"{len(overlap)} patients appear in both {left} and {right}")


def assert_split_integrity(frame: pd.DataFrame) -> None:
    if frame["patient_id"].duplicated().any():
        raise AssertionError("a patient appears more than once in the split manifest")
    _assert_disjoint(frame)
    unknown = set(frame["split"].unique()) - {"train", "validation", "test"}
    if unknown:
        raise AssertionError(f"unknown splits: {unknown}")
