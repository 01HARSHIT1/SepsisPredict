import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import create_app
from backend.app.services.prediction_service import SERVING_MANIFEST, ArtifactError, PredictionService
from src.inference.artifacts import project_root

ROOT = project_root()
MANIFEST = ROOT / SERVING_MANIFEST


pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason="phase 5 serving manifest is not written yet")


@pytest.fixture(scope="module")
def client():
    application = create_app(MANIFEST)
    with TestClient(application) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "risk_probability" not in response.json()
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    local = client.get("/health", headers={"Origin": "http://127.0.0.1:3000"})
    assert local.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"


def test_wildcard_cors_is_rejected(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="wildcard"):
        create_app(MANIFEST)


def test_model_info_matches_manifest_and_hides_paths(client):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    response = client.get("/api/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "research"
    assert body["model_name"] == manifest["model_name"]
    assert body["model_version"] == manifest["model_version"]
    assert body["model_family"] == manifest["model_family"]
    assert body["calibration"]
    assert body["decision_threshold"] == pytest.approx(manifest["decision_threshold"])
    assert body["flexible_input"] == manifest["flexible_input"]
    assert "SepsisLabel" == body["prediction_target"]
    encoded = json.dumps(body)
    assert "artifacts/" not in encoded
    assert "model.json" not in encoded
    assert "C:\\" not in encoded


def test_current_sparse_and_one_variable_predictions_use_the_model(client):
    sparse = client.post("/api/predict", json={"observations": [{"values": {"HR": 118, "MAP": 64, "Resp": 25}}]})
    only_hr = client.post("/api/predict", json={"observations": [{"values": {"HR": 90}}]})
    assert sparse.status_code == 200
    assert only_hr.status_code == 200
    sparse_body = sparse.json()
    hr_body = only_hr.json()
    assert sparse_body["data_available"] == 3
    assert hr_body["data_available"] == 1
    assert hr_body["data_sufficiency"] == "LOW"
    assert sparse_body["data_sufficiency"] == "MODERATE"
    assert sparse_body["risk_trend"] is None
    assert sparse_body["historical_hours"] == 1
    assert 0.0 <= sparse_body["risk_probability"] <= 1.0
    assert 0.0 <= hr_body["risk_probability"] <= 1.0
    assert sparse_body["model_version"] == client.get("/api/model-info").json()["model_version"]
    assert hr_body["explanations"]
    assert hr_body["explanations"][0]["feature"] in client.get("/api/model-info").json()["supported_parameters"]
    assert hr_body["explanations"][0]["direction"] in {"increases_risk_score", "decreases_risk_score"}
    direct = PredictionService.load(manifest_path=MANIFEST).predict_current(
        {"observations": [{"values": {"HR": 90}}]}
    )
    assert hr_body["risk_probability"] == pytest.approx(direct["risk_probability"])


def test_full_input_and_zero_are_not_rewritten(client):
    values = {
        "HR": 88,
        "O2Sat": 97,
        "Temp": 37.0,
        "SBP": 120,
        "MAP": 80,
        "DBP": 60,
        "Resp": 16,
        "Lactate": 1.2,
        "Creatinine": 0.9,
        "WBC": 8.0,
        "Glucose": 110,
        "Age": 64,
        "Gender": 1,
        "ICULOS": 5,
    }
    full = client.post("/api/predict", json={"observations": [{"values": values}]})
    zero = client.post("/api/predict", json={"observations": [{"values": {"HR": 0}}]})
    missing = client.post("/api/predict", json={"observations": [{"values": {"HR": None, "Age": 64}}]})
    assert full.status_code == 200
    assert zero.status_code == 200
    assert missing.status_code == 200
    assert full.json()["data_available"] == len(values)
    assert full.json()["data_sufficiency"] == "HIGH"
    assert zero.json()["data_available"] == 1
    assert missing.json()["data_available"] == 1
    assert missing.json()["data_sufficiency"] == "VERY_LOW"
    assert zero.json()["risk_probability"] != pytest.approx(missing.json()["risk_probability"])


def test_sequence_returns_one_row_per_submitted_hour(client):
    response = client.post(
        "/api/predict-sequence",
        json={
            "observations": [
                {"values": {"HR": 95, "MAP": 82}},
                {"values": {"HR": 101, "MAP": 76}},
                {"values": {"HR": 118, "MAP": 64, "Resp": 25}},
            ]
        },
    )
    assert response.status_code == 200
    rows = response.json()["predictions"]
    assert len(rows) == 3
    assert rows[0]["risk_trend"] is None
    assert rows[2]["historical_hours"] == 3
    assert rows[2]["has_prior_measurements"] is True
    assert rows[0]["research_early_warning"] in {True, False}


def test_unknown_malformed_and_non_chronological_requests(client):
    unknown = client.post("/api/predict", json={"observations": [{"values": {"HeartRate": 90}}]})
    label = client.post("/api/predict", json={"observations": [{"values": {"SepsisLabel": 1}}]})
    text = client.post("/api/predict", json={"observations": [{"values": {"HR": "fast"}}]})
    empty = client.post("/api/predict", json={"observations": []})
    extra = client.post("/api/predict", json={"risk_probability": 0.2, "observations": [{"values": {"HR": 80}}]})
    reversed_hours = client.post(
        "/api/predict-sequence",
        json={"observations": [{"hour": 3, "values": {"HR": 90}}, {"hour": 1, "values": {"HR": 80}}]},
    )
    duplicate = client.post(
        "/api/predict-sequence",
        json={"observations": [{"hour": 1, "values": {"HR": 90}}, {"hour": 1, "values": {"HR": 80}}]},
    )
    many = client.post(
        "/api/predict",
        json={"observations": [{"values": {"HR": 90}}, {"values": {"HR": 100}}]},
    )
    assert unknown.status_code == 422
    assert unknown.json()["error"] == "invalid_input"
    assert label.status_code == 422
    assert text.status_code == 422
    assert empty.status_code == 422
    assert extra.status_code == 422
    assert reversed_hours.status_code == 422
    assert duplicate.status_code == 422
    assert many.status_code == 422
    malformed = client.post("/api/predict", content=b"{", headers={"Content-Type": "application/json"})
    too_many_fields = client.post("/api/predict", json={"observations": [{"values": {f"X{i}": 1 for i in range(41)}}]})
    too_long = client.post(
        "/api/predict-sequence",
        json={"observations": [{"values": {"HR": 80}} for _ in range(337)]},
    )
    mixed_hours = client.post(
        "/api/predict-sequence",
        json={
            "observations": [
                {"values": {"HR": 88}},
                {"values": {"Temp": 38.1, "Lactate": 2.2}},
            ]
        },
    )
    assert malformed.status_code == 422
    assert too_many_fields.status_code == 422
    assert too_long.status_code == 422
    assert "Traceback" not in malformed.text
    assert "Traceback" not in too_many_fields.text
    assert mixed_hours.status_code == 200
    mixed = mixed_hours.json()["predictions"]
    assert [row["data_available"] for row in mixed] == [1, 2]
    assert mixed[0]["data_sufficiency"] == "LOW"
    assert mixed[1]["data_sufficiency"] == "LOW"
    assert mixed[1]["historical_hours"] == 2


def test_all_supported_parameters_remain_a_real_prediction(client):
    names = client.get("/api/model-info").json()["supported_parameters"]
    assert len(names) == 40
    response = client.post("/api/predict", json={"observations": [{"values": {name: 1.0 for name in names}}]})
    assert response.status_code == 200
    body = response.json()
    assert body["data_available"] == 40
    assert body["data_sufficiency"] == "HIGH"
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert "risk_probability" not in client.get("/api/model-info").json()


def test_missing_model_and_version_mismatch(tmp_path: Path):
    with pytest.raises(ArtifactError, match="missing"):
        PredictionService.load(root=tmp_path, manifest_path=tmp_path / "missing.json")
    meta = tmp_path / "artifacts" / "metadata"
    meta.mkdir(parents=True)
    model = tmp_path / "artifacts" / "models"
    model.mkdir(parents=True)
    (model / "model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifacts" / "preprocessors").mkdir()
    (tmp_path / "artifacts" / "preprocessors" / "pre.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifacts" / "calibration").mkdir()
    (tmp_path / "artifacts" / "calibration" / "cal.json").write_text(
        json.dumps({"kind": "isotonic", "thresholds": [0, 1], "values": [0, 1]}),
        encoding="utf-8",
    )
    spec = {
        "features": [
            {"name": "HR_raw", "variable": "HR", "empirical_class": "near_hourly"},
        ]
    }
    (tmp_path / "artifacts" / "feature_definitions").mkdir(parents=True)
    (tmp_path / "artifacts" / "feature_definitions" / "feature_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    metadata = {
        "model_name": "demo",
        "model_version": "v1",
        "expected_feature_order": ["HR_raw"],
        "feature_schema_version": "wrong",
    }
    (meta / "serving.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = {
        "status": "trained",
        "model_path": "artifacts/models/model.json",
        "preprocessor_path": "artifacts/preprocessors/pre.json",
        "calibration_path": "artifacts/calibration/cal.json",
        "metadata_path": "artifacts/metadata/serving.json",
        "feature_spec_path": "artifacts/feature_definitions/feature_spec.json",
        "feature_schema_version": "wrong",
        "model_version": "v2",
        "model_name": "demo",
    }
    manifest_path = meta / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="schema"):
        PredictionService.load(root=tmp_path, manifest_path=manifest_path)
