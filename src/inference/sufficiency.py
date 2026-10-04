"""Technical information band. This is not clinical confidence.

The band uses only the count of vital and laboratory values present at the
scored hour. It does not use the risk probability, the label, or a threshold.
Risk and this band stay separate fields.
"""

from __future__ import annotations

import numpy as np

from src.data.schema import LABS, VITALS
from src.inference.features import DATA_TOTAL, INPUT_INDEX

INFORMATION_BANDS = ("VERY_LOW", "LOW", "MODERATE", "HIGH")
RULE_ID = "dynamic_variables_at_this_hour_v1"
RULE_TEXT = (
    "Count vital and laboratory values that are finite at the scored hour. "
    "HIGH is 6 or more, MODERATE is 3 to 5, LOW is 1 or 2, and VERY_LOW is 0. "
    "Static fields, ICULOS, history length, and the risk probability do not change the band. "
    "This is a data-information label, not clinical confidence and not a diagnosis."
)


def information_descriptor(measurements: np.ndarray, t: int) -> dict:
    row = measurements[t]
    dynamic_index = [INPUT_INDEX[name] for name in (*VITALS, *LABS)]
    laboratory_index = [INPUT_INDEX[name] for name in LABS]
    observed_dynamic = int(np.isfinite(row[dynamic_index]).sum())
    observed_laboratory = int(np.isfinite(row[laboratory_index]).sum())
    present = int(np.isfinite(row).sum())
    prior = False
    if t > 0:
        prior = bool(np.isfinite(measurements[:t][:, dynamic_index]).any())
    return {
        "data_available": present,
        "data_total": DATA_TOTAL,
        "data_completeness": present / DATA_TOTAL,
        "data_sufficiency": band_for_dynamic_count(observed_dynamic),
        "data_sufficiency_rule": RULE_ID,
        "historical_hours": int(t + 1),
        "observed_dynamic_variables": observed_dynamic,
        "observed_laboratory_variables": observed_laboratory,
        "has_prior_measurements": prior,
    }


def band_for_dynamic_count(observed_dynamic: int) -> str:
    if observed_dynamic >= 6:
        return "HIGH"
    if observed_dynamic >= 3:
        return "MODERATE"
    if observed_dynamic >= 1:
        return "LOW"
    return "VERY_LOW"
