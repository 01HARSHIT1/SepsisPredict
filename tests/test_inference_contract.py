import json

import numpy as np
import pandas as pd
import pytest

from src.inference.artifacts import feature_spec_from_audit, model_manifest, write_phase1_artifacts
from src.inference.contract import risk_trend_from_probabilities
from src.inference.features import (
    DATA_TOTAL,
    INPUT_INDEX,
    availability_at,
    build_feature_row,
    parse_observations,
)
from src.inference.service import ModelNotTrainedError, prepare_inference_frame, require_trained_model


def _spec_for(*pairs: tuple[str, str]) -> dict:
    from src.features.temporal import candidate_feature_names

    features = []
    for variable, empirical_class in pairs:
        for name in candidate_feature_names(empirical_class, variable):
            features.append({"name": name, "variable": variable, "empirical_class": empirical_class})
    # The builder indexes the full 40-column matrix, so the spec may list a subset.
    return {"features": features}


def test_omitted_parameter_stays_missing_and_zero_is_kept():
    matrix = parse_observations(
        {"observations": [{"values": {"HR": 0, "MAP": None, "Temp": 39.1}}]}
    )
    assert matrix.shape == (1, DATA_TOTAL)
    assert matrix[0, INPUT_INDEX["HR"]] == 0.0
    assert np.isnan(matrix[0, INPUT_INDEX["MAP"]])
    assert matrix[0, INPUT_INDEX["Temp"]] == pytest.approx(39.1)
    availability = availability_at(matrix, 0)
    assert availability["data_available"] == 2
    assert availability["data_total"] == 40
    assert availability["data_sufficiency"] == "LOW"
    assert availability["observed_dynamic_variables"] == 2
    assert availability["has_prior_measurements"] is False


def test_label_and_unknown_parameter_are_rejected():
    with pytest.raises(ValueError, match="SepsisLabel"):
        parse_observations({"observations": [{"values": {"SepsisLabel": 1}}]})
    with pytest.raises(ValueError, match="unsupported parameter"):
        parse_observations({"observations": [{"values": {"HeartRate": 90}}]})


def test_feature_row_ignores_future_hours_and_does_not_zero_fill():
    spec = _spec_for(("HR", "near_hourly"), ("Lactate", "intermittent"))
    matrix = np.full((4, DATA_TOTAL), np.nan)
    matrix[:, INPUT_INDEX["HR"]] = [80.0, np.nan, 100.0, 140.0]
    matrix[:, INPUT_INDEX["Lactate"]] = [2.4, np.nan, np.nan, 9.0]
    early = build_feature_row(matrix, 2, spec)
    edited = matrix.copy()
    edited[3, INPUT_INDEX["HR"]] = 20.0
    edited[3, INPUT_INDEX["Lactate"]] = 1.0
    later = build_feature_row(edited, 2, spec)
    assert later.keys() == early.keys()
    for key, value in early.items():
        if np.isnan(value):
            assert np.isnan(later[key])
        else:
            assert later[key] == pytest.approx(value)
    assert early["HR_raw"] == pytest.approx(100.0)
    assert np.isnan(early["HR_delta_1h"])
    assert early["Lactate_last"] == pytest.approx(2.4)
    assert early["Lactate_missing"] == 1.0
    assert early["Lactate_time_since_measurement"] == pytest.approx(2.0)
    assert np.isnan(build_feature_row(matrix, 1, spec)["HR_raw"])


def test_static_value_does_not_travel_backward():
    spec = _spec_for(("Age", "static"))
    matrix = np.full((3, DATA_TOTAL), np.nan)
    matrix[2, INPUT_INDEX["Age"]] = 70.0
    assert np.isnan(build_feature_row(matrix, 1, spec)["Age_value"])
    assert build_feature_row(matrix, 1, spec)["Age_missing"] == 1.0
    assert build_feature_row(matrix, 2, spec)["Age_value"] == pytest.approx(70.0)
    carried = matrix.copy()
    carried[0, INPUT_INDEX["Age"]] = 70.0
    assert build_feature_row(carried, 2, spec)["Age_value"] == pytest.approx(70.0)


def test_untrained_manifest_refuses_to_score():
    manifest = model_manifest()
    assert manifest["model_path"] is None
    assert manifest["status"] == "not_trained"
    with pytest.raises(ModelNotTrainedError):
        require_trained_model(manifest)


def test_prepare_frame_has_availability_and_no_probability():
    spec = _spec_for(("HR", "near_hourly"))
    frame = prepare_inference_frame(
        {"observations": [{"values": {"HR": 120}}, {"values": {"HR": 110}}]},
        spec,
    )
    assert "risk_probability" not in frame
    assert frame["hours"][1]["availability"]["data_available"] == 1
    assert frame["hours"][1]["availability"]["historical_hours"] == 2
    assert frame["hours"][1]["features"]["HR_delta_1h"] == pytest.approx(-10.0)


def test_trend_helper_needs_model_probabilities():
    assert risk_trend_from_probabilities([0.2]) is None
    assert risk_trend_from_probabilities([0.2, 0.4]) == "increasing"
    assert risk_trend_from_probabilities([0.4, 0.2]) == "decreasing"
    assert risk_trend_from_probabilities([0.2, 0.2]) == "unchanged"


def test_feature_spec_round_trip(tmp_path):
    table = pd.DataFrame(
        {
            "hospital_system": ["pooled"] * 2,
            "variable": ["HR", "Age"],
            "empirical_class": ["near_hourly", "static"],
        }
    )
    # The exporter requires every input column. Use the saved audit when present,
    # and always check a tiny table fails closed if incomplete.
    with pytest.raises(ValueError, match="missing empirical classes"):
        feature_spec_from_audit(table)
    audit_path = tmp_path / "audit.csv"
    full = pd.read_csv("reports/tables/variable_audit_pooled.csv")
    full.to_csv(audit_path, index=False)
    paths = write_phase1_artifacts(full, root=tmp_path)
    spec = json.loads((tmp_path / "artifacts" / "feature_definitions" / "feature_spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "artifacts" / "metadata" / "model_manifest.json").read_text(encoding="utf-8"))
    assert spec["data_total"] == 40
    assert spec["model_version"] is None
    assert manifest["status"] == "not_trained"
    assert paths["feature_spec"].endswith("feature_spec.json")
    matrix = parse_observations({"observations": [{"values": {"HR": 88}}]})
    row = build_feature_row(matrix, 0, spec)
    assert list(row) == [item["name"] for item in spec["features"]]
    assert row["HR_raw"] == pytest.approx(88.0)
    assert row["MAP_missing"] == 1.0
