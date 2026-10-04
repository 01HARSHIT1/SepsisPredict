"""Serializable artifacts. The model file is separate from the feature definition.

Phase 1 writes the feature definition and a manifest whose model path is empty.
A later training run fills the preprocessor, model, and optional calibrator
without changing the JSON the frontend sends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.temporal import candidate_feature_names
from src.inference.contract import CONTRACT_VERSION, PREDICTION_HORIZON
from src.inference.features import DATA_TOTAL, INPUT_COLUMNS

FEATURE_SPEC_RELATIVE = Path("artifacts") / "feature_definitions" / "feature_spec.json"
MANIFEST_RELATIVE = Path("artifacts") / "metadata" / "model_manifest.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def feature_spec_from_audit(table: pd.DataFrame) -> dict:
    """Freeze feature order from the phase 1 audit table."""
    if "hospital_system" in table.columns:
        pooled = table[table["hospital_system"] == "pooled"]
    else:
        pooled = table
    classes = dict(zip(pooled["variable"], pooled["empirical_class"], strict=False))
    missing = [name for name in INPUT_COLUMNS if name not in classes]
    if missing:
        raise ValueError(f"audit table is missing empirical classes for {missing}")
    features = []
    for name in INPUT_COLUMNS:
        empirical_class = str(classes[name])
        for feature_name in candidate_feature_names(empirical_class, name):
            features.append(
                {
                    "name": feature_name,
                    "variable": name,
                    "empirical_class": empirical_class,
                }
            )
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "feature_definition_only",
        "model_version": None,
        "prediction_horizon": PREDICTION_HORIZON,
        "data_total": DATA_TOTAL,
        "input_variables": list(INPUT_COLUMNS),
        "rejected_inputs": ["SepsisLabel"],
        "missing_value_policy": "Omitted keys and null are missing. Numeric 0 is a measured zero. NaN is never written as 0.",
        "data_sufficiency_policy": None,
        "features": features,
    }


def model_manifest() -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "not_trained",
        "model_version": None,
        "model_family": None,
        "prediction_horizon": PREDICTION_HORIZON,
        "feature_spec_path": FEATURE_SPEC_RELATIVE.as_posix(),
        "preprocessor_path": None,
        "model_path": None,
        "calibration_path": None,
        "metadata_path": MANIFEST_RELATIVE.as_posix(),
        "notes": (
            "Replace model_family and the artifact paths when a model is trained. "
            "The inference service loads these paths. The frontend does not."
        ),
    }


def write_phase1_artifacts(audit_table: pd.DataFrame, root: Path | None = None) -> dict[str, str]:
    root = project_root() if root is None else root
    spec_path = root / FEATURE_SPEC_RELATIVE
    manifest_path = root / MANIFEST_RELATIVE
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    spec = feature_spec_from_audit(audit_table)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(model_manifest(), indent=2), encoding="utf-8")
    for relative in (
        Path("artifacts") / "models",
        Path("artifacts") / "preprocessors",
        Path("artifacts") / "calibration",
    ):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        placeholder = directory / ".gitkeep"
        if not placeholder.exists():
            placeholder.write_text("", encoding="utf-8")
    return {"feature_spec": str(spec_path), "manifest": str(manifest_path)}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
