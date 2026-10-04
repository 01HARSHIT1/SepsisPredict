"""Model-facing entry point. FastAPI will call this and no other feature code.

Any later model family must load through the manifest. This module returns a
probability only after the manifest points at a saved model and preprocessor.
"""

from __future__ import annotations

from pathlib import Path

from src.inference.artifacts import FEATURE_SPEC_RELATIVE, MANIFEST_RELATIVE, load_json, project_root
from src.inference.contract import PREDICTION_HORIZON, risk_trend_from_probabilities
from src.inference.features import availability_at, build_feature_matrix, parse_request
from src.inference.scoring import DISCLAIMER, load_serving_bundle, score_feature_matrix


class ModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested before a model artifact exists."""


def require_trained_model(manifest: dict) -> None:
    if manifest.get("status") != "trained" or not manifest.get("model_path"):
        raise ModelNotTrainedError(
            "No trained model artifact is available. "
            "A probability is returned only after training writes the manifest."
        )


def load_manifest(root: Path | None = None) -> dict:
    root = project_root() if root is None else root
    return load_json(root / MANIFEST_RELATIVE)


def prepare_inference_frame(payload: dict | list, spec: dict) -> dict:
    """Parse a request and build per-hour features plus availability.

    Does not score a model. Gap hours are used to build features and are not
    returned. ``risk_probability`` is not part of this result.
    """
    measurements, scored, hours = parse_request(payload)
    rows = build_feature_matrix(measurements, spec)
    frame = []
    for position, hour_index in enumerate(scored):
        index = int(hour_index)
        frame.append(
            {
                "hour_index": index,
                "hour": hours[position],
                "features": rows[index],
                "availability": availability_at(measurements, index),
            }
        )
    return {"hours": frame}


def predict(
    payload: dict | list,
    root: Path | None = None,
    manifest_path: Path | str | None = None,
    manifest: dict | None = None,
    bundle: dict | None = None,
    spec: dict | None = None,
) -> dict:
    """Score a request with the model named by the manifest.

    A caller that already loaded the booster, such as the API process, can
    pass ``manifest``, ``bundle``, and ``spec`` so the files are not read again.
    """
    root = project_root() if root is None else root
    if manifest is None:
        if manifest_path is None:
            manifest = load_manifest(root)
        else:
            path = Path(manifest_path)
            if not path.is_absolute():
                path = root / path
            manifest = load_json(path)
    require_trained_model(manifest)
    if spec is None:
        spec = load_json(root / manifest.get("feature_spec_path", FEATURE_SPEC_RELATIVE.as_posix()))
    frame = prepare_inference_frame(payload, spec)
    names = [item["name"] for item in spec["features"]]
    if manifest.get("model_family") == "gru":
        from src.inference.gru_runtime import score_gru_payload

        probabilities = score_gru_payload(payload, manifest, root, spec)
        explanations = [[] for _ in probabilities]
        metadata = load_json(root / manifest["metadata_path"])
    else:
        matrix = _frame_matrix(frame, names)
        if bundle is None:
            bundle = load_serving_bundle(manifest, root)
        probabilities, explanations = score_feature_matrix(matrix, names, bundle)
        metadata = bundle["metadata"]
    threshold = float(metadata["decision_threshold"])
    probability_list = [float(value) for value in probabilities]
    predictions = []
    for position, hour in enumerate(frame["hours"]):
        trend_source = probability_list[: position + 1]
        availability = hour["availability"]
        probability = probability_list[position]
        predictions.append(
            {
                "hour": hour["hour"],
                "risk_probability": probability,
                "research_early_warning": bool(probability >= threshold),
                "data_available": availability["data_available"],
                "data_total": availability["data_total"],
                "data_completeness": availability["data_completeness"],
                "data_sufficiency": availability["data_sufficiency"],
                "data_sufficiency_rule": availability["data_sufficiency_rule"],
                "historical_hours": availability["historical_hours"],
                "observed_dynamic_variables": availability["observed_dynamic_variables"],
                "observed_laboratory_variables": availability["observed_laboratory_variables"],
                "has_prior_measurements": availability["has_prior_measurements"],
                "risk_trend": risk_trend_from_probabilities(trend_source),
                "explanations": explanations[position],
            }
        )
    return {
        "contract_version": manifest["contract_version"],
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "model_family": manifest["model_family"],
        "prediction_target": metadata["prediction_target"],
        "prediction_horizon": PREDICTION_HORIZON,
        "decision_threshold": threshold,
        "data_sufficiency": predictions[-1]["data_sufficiency"] if predictions else None,
        "disclaimer": DISCLAIMER,
        "predictions": predictions,
    }


def _frame_matrix(frame: dict, names: list[str]) -> "object":
    import numpy as np

    rows = []
    for hour in frame["hours"]:
        rows.append([hour["features"][name] for name in names])
    return np.asarray(rows, dtype=np.float64)
