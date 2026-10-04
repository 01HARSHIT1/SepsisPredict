"""Imputation hypotheses. Nothing here fills a matrix.

Phase 1 only recommends a strategy from measured frequency. Whether a strategy
helps is a later validation experiment. Zero-filling is not a recommendation.
"""

from __future__ import annotations


def empirical_frequency_class(role: str, pct_hours_observed: float, median_gap_hours: float | None) -> str:
    if role == "static":
        return "static"
    if role == "time_context":
        return "time_context"
    if role == "target":
        return "target"
    near_hourly = pct_hours_observed >= 40.0 and median_gap_hours is not None and median_gap_hours <= 2
    if near_hourly:
        return "near_hourly"
    intermittent = pct_hours_observed >= 5.0 or (median_gap_hours is not None and median_gap_hours <= 12)
    if intermittent:
        return "intermittent"
    return "sparse"


def recommend_representation(role: str, empirical_class: str, pct_patients_with_value_change: float) -> str:
    if role == "target":
        return (
            "Do not impute and do not use as an input. SepsisLabel is the challenge target, "
            "already shifted to 6 hours before the reconstructed onset."
        )
    if role == "time_context":
        return (
            "Use ICULOS as the current hour index. Do not build physiologic deltas from it "
            "and do not impute it; it is structural."
        )
    if role == "static" or empirical_class == "static":
        return (
            "Carry a constant within-patient value across hours only when that value is already "
            "present at the first hour of the stay. If the whole stay is missing, leave the value "
            "missing and add a missing indicator. Do not fill with 0. Do not compute hourly deltas."
        )
    if empirical_class == "near_hourly" and pct_patients_with_value_change >= 50:
        return (
            "Keep the raw hour-t measurement and a missing indicator. Candidate backward features "
            "(previous value, 1-3 hour deltas, 3-6 hour mean/spread/slope) are worth testing because "
            "the series is both frequent and usually changing. A one-hour backward carry is a "
            "validation hypothesis only. Do not back-fill from the future and do not zero-fill."
        )
    if empirical_class == "near_hourly":
        return (
            "The series is frequently present but often numerically unchanged. Keep raw value and "
            "missingness. Do not automatically add slopes. Do not zero-fill or back-fill."
        )
    if empirical_class == "intermittent":
        return (
            "Treat as an irregular measurement. Store the current raw value separately from the last "
            "real value, with time since that measurement and a missing indicator. A 6-hour mean of "
            "real measurements may be tested. Do not forward-fill across a long gap as if the lab "
            "had been redrawn."
        )
    return (
        "Sparse measurement. Do not forward-fill as a current value. Represent last observed value, "
        "whether the current hour has a real measurement, time since that measurement, change between "
        "the last two real measurements when both exist, and the count of real measurements in a "
        "recent window. Leave last value missing until the first real measurement."
    )
