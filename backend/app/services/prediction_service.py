"""Load the Phase 5 serving artifacts once and score through the shared inference path."""

from __future__ import annotations

from pathlib import Path

from src.inference.artifacts import load_json, project_root
from src.inference.scoring import DISCLAIMER, load_serving_bundle
from src.inference.service import predict, require_trained_model
from src.inference.sufficiency import RULE_ID, RULE_TEXT
from src.training.train import feature_schema_version

SERVING_MANIFEST = Path("artifacts") / "metadata" / "phase5_model_manifest.json"


class ArtifactError(RuntimeError):
    """Raised when the serving manifest or a required artifact is unusable."""


class PredictionService:
    def __init__(self, manifest: dict, spec: dict, bundle: dict):
        self.manifest = manifest
        self.spec = spec
        self.bundle = bundle

    @classmethod
    def load(cls, root: Path | None = None, manifest_path: Path | str | None = None) -> "PredictionService":
        root = project_root() if root is None else root
        path = root / SERVING_MANIFEST if manifest_path is None else Path(manifest_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ArtifactError("serving manifest is missing")
        manifest = load_json(path)
        try:
            require_trained_model(manifest)
        except RuntimeError as exc:
            raise ArtifactError(str(exc)) from exc
        for key in ("model_path", "preprocessor_path", "metadata_path", "feature_spec_path"):
            if not manifest.get(key):
                raise ArtifactError(f"manifest is missing {key}")
            if not (root / manifest[key]).is_file():
                raise ArtifactError(f"required artifact is missing: {key}")
        if not manifest.get("calibration_path"):
            raise ArtifactError("manifest is missing calibration_path")
        if not (root / manifest["calibration_path"]).is_file():
            raise ArtifactError("required artifact is missing: calibration_path")
        spec = load_json(root / manifest["feature_spec_path"])
        schema = feature_schema_version([item["name"] for item in spec["features"]])
        if manifest.get("feature_schema_version") != schema:
            raise ArtifactError("feature schema does not match the manifest")
        metadata = load_json(root / manifest["metadata_path"])
        if metadata.get("model_version") != manifest.get("model_version"):
            raise ArtifactError("model version does not match the manifest")
        if metadata.get("model_name") != manifest.get("model_name"):
            raise ArtifactError("model name does not match the manifest")
        if metadata.get("feature_schema_version") != schema:
            raise ArtifactError("feature schema does not match the model metadata")
        bundle = load_serving_bundle(manifest, root)
        if bundle["calibrator"] is None:
            raise ArtifactError("calibration artifact did not load")
        return cls(manifest, spec, bundle)

    def model_info(self) -> dict:
        metadata = self.bundle["metadata"]
        return {
            "status": self.manifest.get("deployment_status", "research"),
            "model_name": metadata["model_name"],
            "model_version": metadata["model_version"],
            "model_family": self.manifest["model_family"],
            "prediction_target": metadata["prediction_target"],
            "prediction_horizon": metadata["prediction_horizon"],
            "supported_parameters": list(metadata["supported_input_variables"]),
            "flexible_input": bool(metadata["flexible_input"]),
            "calibration": metadata["calibration_kind"],
            "decision_threshold": float(metadata["decision_threshold"]),
            "threshold_label": "research_early_warning",
            "training_dataset_version": metadata["training_dataset_version"],
            "feature_schema_version": metadata["feature_schema_version"],
            "data_sufficiency_rule": RULE_ID,
            "data_sufficiency_description": RULE_TEXT,
            "disclaimer": DISCLAIMER,
        }

    def predict_current(self, payload: dict) -> dict:
        observations = payload.get("observations", [])
        if len(observations) != 1:
            raise ValueError("predict accepts one current observation; use predict-sequence for history")
        result = self._score(payload)
        hour = result["predictions"][0]
        return {
            "risk_probability": hour["risk_probability"],
            "research_early_warning": hour["research_early_warning"],
            "data_available": hour["data_available"],
            "data_total": hour["data_total"],
            "data_completeness": hour["data_completeness"],
            "data_sufficiency": hour["data_sufficiency"],
            "data_sufficiency_rule": hour["data_sufficiency_rule"],
            "historical_hours": hour["historical_hours"],
            "observed_dynamic_variables": hour["observed_dynamic_variables"],
            "observed_laboratory_variables": hour["observed_laboratory_variables"],
            "has_prior_measurements": hour["has_prior_measurements"],
            "risk_trend": hour["risk_trend"],
            "prediction_horizon": result["prediction_horizon"],
            "model_version": result["model_version"],
            "model_name": result["model_name"],
            "explanations": hour["explanations"],
            "disclaimer": result["disclaimer"],
        }

    def predict_sequence(self, payload: dict) -> dict:
        result = self._score(payload)
        return {
            "model_name": result["model_name"],
            "model_version": result["model_version"],
            "prediction_horizon": result["prediction_horizon"],
            "disclaimer": result["disclaimer"],
            "predictions": result["predictions"],
        }

    def _score(self, payload: dict) -> dict:
        return predict(payload, manifest=self.manifest, bundle=self.bundle, spec=self.spec)
