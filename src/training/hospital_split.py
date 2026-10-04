"""Patient sets for cross-hospital experiments.

Training and validation stay inside the source hospital. The other hospital
is used only as the external test set. A patient cannot be in both.
"""

from __future__ import annotations

import pandas as pd


def patient_ids_for(assignments: pd.DataFrame, hospital: str, split: str | None = None) -> set[str]:
    frame = assignments[assignments["hospital_system"] == hospital]
    if split is not None:
        frame = frame[frame["split"] == split]
    return set(frame["patient_id"].astype(str))


def assert_patient_disjoint(left: set[str], right: set[str]) -> None:
    overlap = left & right
    if overlap:
        raise ValueError(f"patient overlap: {len(overlap)}")
