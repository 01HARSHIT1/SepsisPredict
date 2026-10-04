"""JSON contract for a later FastAPI service.

The shapes below are the input and output the frontend will exchange with Python.
This module does not start a server and does not fill in a risk probability.
"""

from __future__ import annotations

CONTRACT_VERSION = "1"

# Official early-warning label. Not a confirmed diagnosis at the current hour.
PREDICTION_HORIZON = {
    "name": "official_sepsis_label",
    "description": (
        "Hourly probability of the PhysioNet 2019 SepsisLabel. "
        "For a patient who meets the challenge definition, that label is 1 from "
        "t_sepsis - 6 hours onward."
    ),
}

# Paths a trained release will fill. Null means the artifact does not exist yet.
ARTIFACT_SLOTS = (
    "feature_spec",
    "preprocessor",
    "model",
    "calibration",
    "metadata",
)


def risk_trend_from_probabilities(probabilities: list[float]) -> str | None:
    """Compare the latest model probability with the previous hour.

    Returns None when fewer than two model probabilities exist.
    The caller must pass probabilities emitted by a loaded model.
    """
    if len(probabilities) < 2:
        return None
    latest = float(probabilities[-1])
    previous = float(probabilities[-2])
    if latest > previous:
        return "increasing"
    if latest < previous:
        return "decreasing"
    return "unchanged"


def empty_prediction_slot() -> dict:
    """Response fields a later endpoint will populate. Risk stays absent."""
    return {
        "risk_probability": None,
        "data_available": None,
        "data_total": None,
        "data_completeness": None,
        "data_sufficiency": None,
        "risk_trend": None,
        "prediction_horizon": PREDICTION_HORIZON["name"],
        "model_version": None,
        "explanations": [],
    }
