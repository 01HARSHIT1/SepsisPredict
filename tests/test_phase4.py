import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.alarms import false_alarm_summary, lead_time_summary
from src.inference.service import predict
from src.models.gru_inputs import fit_column_scaler, to_network_inputs
from src.models.gru_net import HourlyGRU
from src.models.sequences import patient_block, window_coverage, windows_ending_at_each_hour
from src.training.flexible_mask import (
    TRAIN_SEED,
    apply_drop_mask,
    keep_variables_drop,
    layout_from_spec,
    training_drop_mask,
)
from src.training.hospital_split import assert_patient_disjoint, patient_ids_for


def _tiny_spec():
    return {
        "features": [
            {"name": "HR_raw", "variable": "HR", "empirical_class": "near_hourly"},
            {"name": "HR_missing", "variable": "HR", "empirical_class": "near_hourly"},
            {"name": "Temp_raw", "variable": "Temp", "empirical_class": "intermittent"},
            {"name": "Temp_missing", "variable": "Temp", "empirical_class": "intermittent"},
            {"name": "Lactate_raw", "variable": "Lactate", "empirical_class": "intermittent"},
            {"name": "Lactate_missing", "variable": "Lactate", "empirical_class": "intermittent"},
            {"name": "Lactate_last", "variable": "Lactate", "empirical_class": "intermittent"},
            {"name": "Lactate_time_since_measurement", "variable": "Lactate", "empirical_class": "intermittent"},
            {"name": "Lactate_measurements_in_24h", "variable": "Lactate", "empirical_class": "intermittent"},
            {"name": "Age_value", "variable": "Age", "empirical_class": "static"},
            {"name": "Age_missing", "variable": "Age", "empirical_class": "static"},
            {"name": "ICULOS", "variable": "ICULOS", "empirical_class": "time_context"},
        ]
    }


def test_training_mask_is_deterministic_and_keeps_a_measured_zero():
    spec = json.loads(open("artifacts/feature_definitions/feature_spec.json", encoding="utf-8").read())
    layout = layout_from_spec(spec)
    rates = np.linspace(0.1, 1.0, 40)
    ranked = list(layout.variables)
    rows = np.arange(8)
    first = training_drop_mask(rows, rates, ranked, TRAIN_SEED)
    second = training_drop_mask(rows, rates, ranked, TRAIN_SEED)
    assert np.array_equal(first, second)
    block = np.zeros((1, len(layout.names)), dtype=np.float32)
    hr_raw = layout.names.index("HR_raw")
    hr_missing = layout.names.index("HR_missing")
    block[0, hr_raw] = 0.0
    block[0, hr_missing] = 0.0
    kept = apply_drop_mask(block, np.zeros((1, 40), dtype=bool), layout)
    assert kept[0, hr_raw] == pytest.approx(0.0)
    assert kept[0, hr_missing] == pytest.approx(0.0)
    dropped = keep_variables_drop(1, {"Temp"})
    blanked = apply_drop_mask(block, dropped, layout)
    assert np.isnan(blanked[0, hr_raw])
    assert blanked[0, hr_missing] == pytest.approx(1.0)


def test_one_variable_mask_and_sparse_lab_are_distinct_from_zero():
    spec = json.loads(open("artifacts/feature_definitions/feature_spec.json", encoding="utf-8").read())
    layout = layout_from_spec(spec)
    drop = keep_variables_drop(1, {"Lactate"})
    assert drop[0, layout.variables.index("HR")]
    assert not drop[0, layout.variables.index("Lactate")]
    names = ["Lactate_raw", "Lactate_missing", "Lactate_last", "Lactate_time_since_measurement"]
    values = np.array(
        [[np.nan, 1.0, np.nan, np.nan], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    scaler = fit_column_scaler(values, names)
    network = to_network_inputs(values, names, scaler)
    assert network[0, 1] == pytest.approx(1.0)
    assert network[1, 1] == pytest.approx(0.0)
    assert not np.array_equal(network[0], network[1])


def test_windows_do_not_read_the_future_and_patients_stay_isolated():
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    pad = np.array([-1.0, -1.0], dtype=np.float32)
    original = windows_ending_at_each_hour(values, 3, pad)
    edited = values.copy()
    edited[-1] = 99
    changed = windows_ending_at_each_hour(edited, 3, pad)
    assert np.array_equal(original[:-1], changed[:-1])
    coverage = window_coverage(np.array([4, 8]), 6)
    assert coverage["hours_with_left_padding"] == 4 + 5
    features = np.array([[1.0], [1.0], [7.0], [7.0]])
    block = patient_block(features, np.array([0, 2, 4]), 0)
    assert np.all(block == 1.0)


def test_lead_time_bins_and_false_alarm_episodes():
    labels = np.zeros(20)
    labels[10:] = 1
    probabilities = np.zeros(20)
    probabilities[4] = 0.9
    summary = lead_time_summary(labels, probabilities, np.array([0, 20]), 0.5)
    assert summary["lead_hours"]["median"] == pytest.approx(12.0)
    assert summary["bins_among_onset_inside"]["from_12h_to_24h_before_onset"] == 1
    quiet = np.zeros(48)
    alarms = np.zeros(48)
    alarms[:3] = 1
    alarms[10] = 1
    false_alarms = false_alarm_summary(quiet, alarms, np.array([0, 48]), 0.5)
    assert false_alarms["false_positive_hours"] == 4
    assert false_alarms["mean_false_alarm_episode_hours"] == pytest.approx(2.0)
    assert false_alarms["n_persistent_false_alarm_episodes"] == 1
    assert false_alarms["false_positive_hours_per_nonseptic_patient_day"] == pytest.approx(2.0)


def test_hospital_partitions_do_not_share_patients():
    frame = pd.DataFrame(
        {
            "patient_id": ["a1", "a2", "b1"],
            "hospital_system": ["A", "A", "B"],
            "split": ["train", "validation", "test"],
        }
    )
    source = patient_ids_for(frame, "A", "train") | patient_ids_for(frame, "A", "validation")
    external = patient_ids_for(frame, "B")
    assert_patient_disjoint(source, external)
    with pytest.raises(ValueError, match="overlap"):
        assert_patient_disjoint(source, source)


def test_gru_manifest_loads_threshold_calibration_and_one_input(tmp_path):
    spec = _tiny_spec()
    names = ["HR_raw", "HR_missing"]
    model = HourlyGRU(2, hidden_size=4)
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.constant_(model.head.bias, -1.0)
    state_path = tmp_path / "gru.pt"
    torch.save(model.state_dict(), state_path)
    (tmp_path / "artifacts" / "feature_definitions").mkdir(parents=True)
    (tmp_path / "artifacts" / "preprocessors").mkdir(parents=True)
    (tmp_path / "artifacts" / "calibration").mkdir(parents=True)
    (tmp_path / "artifacts" / "metadata").mkdir(parents=True)
    (tmp_path / "artifacts" / "feature_definitions" / "feature_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    config = {
        "kind": "gru_window",
        "window": 2,
        "hidden_size": 4,
        "feature_names": names,
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "state_dict_path": "gru.pt",
    }
    (tmp_path / "artifacts" / "preprocessors" / "gru.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "artifacts" / "calibration" / "gru.json").write_text(
        json.dumps({"kind": "isotonic", "thresholds": [0.0, 1.0], "values": [0.0, 1.0]}),
        encoding="utf-8",
    )
    metadata = {
        "model_name": "gru_window_2",
        "model_version": "phase4-test",
        "model_family": "gru",
        "training_dataset_version": "physionet-challenge-2019-1.0.0",
        "feature_schema_version": "features-test",
        "prediction_target": "SepsisLabel",
        "prediction_horizon": {"name": "official_sepsis_label", "description": "test"},
        "training_timestamp": "2026-01-01T00:00:00+00:00",
        "expected_feature_order": names,
        "decision_threshold": 0.42,
        "flexible_input": False,
        "supported_input_variables": ["HR"],
    }
    (tmp_path / "artifacts" / "metadata" / "gru.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = {
        "contract_version": "1",
        "status": "trained",
        "model_version": "phase4-test",
        "model_family": "gru",
        "model_name": "gru_window_2",
        "model_path": "gru.pt",
        "feature_spec_path": "artifacts/feature_definitions/feature_spec.json",
        "preprocessor_path": "artifacts/preprocessors/gru.json",
        "calibration_path": "artifacts/calibration/gru.json",
        "metadata_path": "artifacts/metadata/gru.json",
        "decision_threshold": 0.42,
        "flexible_input": False,
    }
    manifest_path = tmp_path / "artifacts" / "metadata" / "phase4_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = predict({"HR": 90}, root=tmp_path, manifest_path=manifest_path)
    assert result["predictions"][0]["data_available"] == 1
    assert result["predictions"][0]["data_sufficiency"] == "LOW"
    assert 0.0 < result["predictions"][0]["risk_probability"] < 1.0
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["decision_threshold"] == pytest.approx(0.42)
    assert json.loads((tmp_path / "artifacts" / "calibration" / "gru.json").read_text(encoding="utf-8"))["kind"] == "isotonic"
