import json

import numpy as np
import pytest

from src.features.assemble import assemble_patient
from src.inference.features import INPUT_INDEX, parse_request
from src.inference.scoring import apply_calibrator, apply_preprocessor, sigmoid
from src.inference.service import predict
from src.training.masks import mask_definitions, rank_variables


def test_flat_request_keeps_omitted_fields_missing():
    measurements, scored, hours = parse_request({"HR": 118, "Temp": 38.7, "MAP": 64, "Resp": 25})
    assert measurements.shape == (1, 40)
    assert scored.tolist() == [0]
    assert hours == [None]
    assert measurements[0, INPUT_INDEX["HR"]] == pytest.approx(118)
    assert np.isnan(measurements[0, INPUT_INDEX["O2Sat"]])
    assert measurements[0, INPUT_INDEX["Resp"]] == pytest.approx(25)


def test_hour_gap_does_not_create_a_one_hour_delta():
    spec = {
        "features": [
            {"name": "HR_raw", "variable": "HR", "empirical_class": "near_hourly"},
            {"name": "HR_delta_1h", "variable": "HR", "empirical_class": "near_hourly"},
        ]
    }
    measurements, scored, hours = parse_request(
        [{"hour": 1, "HR": 80}, {"hour": 3, "HR": 100}]
    )
    assert measurements.shape == (3, 40)
    assert np.isnan(measurements[1, INPUT_INDEX["HR"]])
    assert scored.tolist() == [0, 2]
    assert hours == [1, 3]
    assembled = assemble_patient(measurements, spec)
    assert np.isnan(assembled[2, 1])
    assert assembled[2, 0] == pytest.approx(100)


def test_future_edit_does_not_change_an_earlier_assembled_row():
    spec = json.loads(open("artifacts/feature_definitions/feature_spec.json", encoding="utf-8").read())
    rng = np.random.default_rng(7)
    measurements = rng.normal(size=(8, 40))
    measurements[rng.random(measurements.shape) < 0.5] = np.nan
    original = assemble_patient(measurements, spec)
    edited = measurements.copy()
    edited[7, INPUT_INDEX["HR"]] = 180
    edited[7, INPUT_INDEX["Lactate"]] = 12
    changed = assemble_patient(edited, spec)
    assert np.allclose(original[:7], changed[:7], equal_nan=True)


def test_count_mask_breaks_observation_ties_by_column_order():
    counts = np.zeros(40)
    counts[0] = 10
    counts[1] = 10
    ranked = rank_variables(counts, n_hours=10)
    assert ranked[0]["variable"] == "HR"
    assert ranked[1]["variable"] == "O2Sat"
    spec = {
        "features": [
            {"name": "HR_raw", "variable": "HR", "empirical_class": "near_hourly"},
            {"name": "Age_value", "variable": "Age", "empirical_class": "static"},
            {"name": "Age_missing", "variable": "Age", "empirical_class": "static"},
        ]
    }
    masks = {item["name"]: item for item in mask_definitions(spec, ranked)}
    assert masks["hr_only"]["variables"] == ["HR"]
    assert masks["top_1"]["variables"] == ["HR"]
    assert "Age_value" in masks["static_and_time"]["feature_names"]
    assert "HR_raw" not in masks["labs_only"]["feature_names"]


def test_logistic_artifacts_round_trip(tmp_path):
    spec = {
        "contract_version": "1",
        "features": [
            {"name": "HR_raw", "variable": "HR", "empirical_class": "near_hourly"},
        ],
    }
    (tmp_path / "artifacts" / "feature_definitions").mkdir(parents=True)
    (tmp_path / "artifacts" / "models").mkdir()
    (tmp_path / "artifacts" / "preprocessors").mkdir()
    (tmp_path / "artifacts" / "calibration").mkdir()
    (tmp_path / "artifacts" / "metadata").mkdir()
    (tmp_path / "artifacts" / "feature_definitions" / "feature_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    preprocessor = {
        "kind": "median_imputer_standard_scaler",
        "feature_names": ["HR_raw"],
        "medians": [0.0],
        "mean": [0.0],
        "scale": [1.0],
    }
    (tmp_path / "artifacts" / "preprocessors" / "logistic_unweighted.json").write_text(
        json.dumps(preprocessor), encoding="utf-8"
    )
    np.savez(
        tmp_path / "artifacts" / "models" / "logistic_unweighted.npz",
        coefficients=np.array([0.1]),
        intercept=np.array([-2.0]),
    )
    (tmp_path / "artifacts" / "calibration" / "serving_calibrator.json").write_text(
        json.dumps({"kind": "isotonic", "thresholds": [0.0, 1.0], "values": [0.0, 1.0]}),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "metadata" / "serving_model.json").write_text(
        json.dumps(
            {
                "model_name": "logistic_unweighted",
                "model_version": "test",
                "model_family": "logistic_regression",
                "training_dataset_version": "physionet-challenge-2019-1.0.0",
                "feature_schema_version": "features-test",
                "prediction_target": "SepsisLabel",
                "prediction_horizon": {"name": "official_sepsis_label", "description": "test"},
                "training_timestamp": "2026-01-01T00:00:00+00:00",
                "expected_feature_order": ["HR_raw"],
                "decision_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "metadata" / "model_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": "1",
                "status": "trained",
                "model_version": "test",
                "model_family": "logistic_regression",
                "model_name": "logistic_unweighted",
                "prediction_horizon": {"name": "official_sepsis_label"},
                "feature_spec_path": "artifacts/feature_definitions/feature_spec.json",
                "preprocessor_path": "artifacts/preprocessors/logistic_unweighted.json",
                "model_path": "artifacts/models/logistic_unweighted.npz",
                "calibration_path": "artifacts/calibration/serving_calibrator.json",
                "metadata_path": "artifacts/metadata/serving_model.json",
            }
        ),
        encoding="utf-8",
    )
    result = predict({"HR": 80}, root=tmp_path)
    expected = float(sigmoid(np.array([0.1 * 80 - 2.0]))[0])
    assert result["predictions"][0]["risk_probability"] == pytest.approx(expected)
    assert result["predictions"][0]["data_available"] == 1
    assert result["predictions"][0]["data_sufficiency"] == "LOW"
    assert result["predictions"][0]["explanations"][0]["feature"] == "HR"
    assert result["predictions"][0]["explanations"][0]["direction"] == "increases_risk_score"
    assert result["data_sufficiency"] == "LOW"


def test_preprocessor_leaves_tree_missing_values_missing():
    block = np.array([[1.0, np.nan]], dtype=np.float32)
    transformed, _inputs = apply_preprocessor(
        block, {"kind": "native_missing", "feature_names": ["a", "b"]}
    )
    assert transformed[0, 0] == pytest.approx(1)
    assert np.isnan(transformed[0, 1])
    calibrated = apply_calibrator(
        np.array([0.2, 0.8]),
        {"kind": "isotonic", "thresholds": [0.0, 1.0], "values": [0.1, 0.9]},
    )
    assert calibrated[0] == pytest.approx(0.26)
