import numpy as np
import pandas as pd
import pytest

from src.data.schema import EXPECTED_COLUMNS, TARGET
from src.data.audit import CohortAuditor
from src.data.splits import assign_patient_splits
from src.labels.targets import future_positive_within_horizon, reconstructed_onset_index


def _blank_stay(n_hours: int) -> np.ndarray:
    values = np.full((n_hours, len(EXPECTED_COLUMNS)), np.nan)
    iculos = EXPECTED_COLUMNS.index("ICULOS")
    label = EXPECTED_COLUMNS.index(TARGET)
    values[:, iculos] = np.arange(1, n_hours + 1)
    values[:, label] = 0
    return values


def test_nan_heart_rate_is_not_averaged_as_zero():
    auditor = CohortAuditor()
    stay = _blank_stay(3)
    hr = EXPECTED_COLUMNS.index("HR")
    stay[:, hr] = [80.0, np.nan, 100.0]
    auditor.add_patient("A", "p1", stay)
    result = auditor.finalize()
    row = result.variable_table[
        (result.variable_table["variable"] == "HR") & (result.variable_table["hospital_system"] == "pooled")
    ].iloc[0]
    assert row["n_observed"] == 2
    assert row["mean"] == pytest.approx(90.0)
    assert row["median_gap_hours"] == pytest.approx(2.0)


def test_split_is_disjoint_and_complete():
    patients = pd.DataFrame(
        {
            "patient_id": [f"p{i:03d}" for i in range(100)],
            "hospital_system": ["A" if i < 60 else "B" for i in range(100)],
            "sepsis_patient": [1 if i % 5 == 0 else 0 for i in range(100)],
        }
    )
    result = assign_patient_splits(patients, seed=2019)
    assert result.summary["overlap_patient_count"] == 0
    assert result.summary["n_train"] + result.summary["n_validation"] + result.summary["n_test"] == 100
    counts = result.assignments["split"].value_counts()
    assert counts["train"] == result.summary["n_train"]


def test_onset_reconstruction_matches_challenge_shift():
    labels = np.array([0, 0, 0, 1, 1])
    # First positive index is 3, challenge onset index is 3 + 6 = 9, past the end of this file.
    assert reconstructed_onset_index(labels) == 9


def test_future_horizon_target_does_not_count_the_current_hour():
    labels = np.array([0, 0, 1, 1])
    assert future_positive_within_horizon(labels, t=0, horizon_hours=1) == 0
    assert future_positive_within_horizon(labels, t=0, horizon_hours=2) == 1
    assert future_positive_within_horizon(labels, t=2, horizon_hours=1) == 1
