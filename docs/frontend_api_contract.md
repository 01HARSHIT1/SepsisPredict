# Frontend API contract

The Next.js client in `frontend/` sends raw clinical values and displays the JSON response. It does not build model features, calibrate probabilities, or decide data sufficiency.

The service is a research early-warning model for the PhysioNet 2019 `SepsisLabel`. It is not a diagnosis system. `research_early_warning` means the calibrated probability is at or above the validation threshold stored with the model. It is not a diagnosis.

## Base URL

Local development:

```
http://127.0.0.1:8000
```

Start it from the repository root:

```
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Endpoints

| method | path | purpose |
| --- | --- | --- |
| GET | `/health` | Process is up |
| GET | `/api/model-info` | Active research model |
| POST | `/api/predict` | One current observation |
| POST | `/api/predict-sequence` | Chronological observations, one score per submitted hour |

## Request

Current observation:

```json
{
  "observations": [
    {
      "values": {
        "HR": 118,
        "Temp": 38.7,
        "MAP": 64,
        "Resp": 25
      }
    }
  ]
}
```

History, in chronological order. Hours may be omitted when the array itself is the order:

```json
{
  "observations": [
    {"values": {"HR": 95, "MAP": 82}},
    {"values": {"HR": 101, "MAP": 76}},
    {"values": {"HR": 118, "MAP": 64, "Resp": 25}}
  ]
}
```

Explicit hours must be unique and already sorted:

```json
{
  "observations": [
    {"hour": 1, "values": {"HR": 95}},
    {"hour": 3, "values": {"HR": 118, "Resp": 25}}
  ]
}
```

The gap hour is used only to keep the timeline honest. It is not returned as a prediction.

Rules:

- Omitted field: unavailable
- JSON null: unavailable
- Numeric 0: a measured zero
- Unknown parameter name: rejected
- `SepsisLabel`: rejected
- Non-numeric clinical value: rejected
- Duplicate `hour`: rejected
- Hours that are not chronological: rejected
- `/api/predict` accepts exactly one observation
- A request may contain at most 336 observations, and each observation may contain at most 40 values
- The client does not send engineered features, a risk probability, or a data-sufficiency label

Abnormal clinical numbers are still valid input. A heart rate of 140 is not an invalid request. A heart rate of `"high"` is an invalid request.

Supported parameters are the 40 PhysioNet 2019 inputs: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.

`GET /api/model-info` returns the list the loaded model actually supports.

## Responses

`GET /health`

```json
{"status": "ok"}
```

`GET /api/model-info` includes `status` of `research`, model name, model version, family, target, prediction horizon, supported parameters, whether the model was trained as a flexible-input model, calibration kind, decision threshold, dataset version, feature schema version, the data-sufficiency rule, and the disclaimer. It does not include filesystem paths.

`POST /api/predict`

```json
{
  "risk_probability": 0.02,
  "research_early_warning": false,
  "data_available": 4,
  "data_total": 40,
  "data_completeness": 0.1,
  "data_sufficiency": "MODERATE",
  "data_sufficiency_rule": "dynamic_variables_at_this_hour_v1",
  "historical_hours": 1,
  "observed_dynamic_variables": 4,
  "observed_laboratory_variables": 0,
  "has_prior_measurements": false,
  "risk_trend": null,
  "prediction_horizon": {
    "name": "official_sepsis_label",
    "description": "Hourly probability of the PhysioNet 2019 SepsisLabel."
  },
  "model_version": "phase5-...",
  "model_name": "...",
  "explanations": [
    {
      "feature": "HR",
      "contribution": 0.1,
      "direction": "increases_risk_score",
      "contribution_space": "log_odds_before_calibration"
    }
  ],
  "disclaimer": "Research estimate of the PhysioNet 2019 early-warning label. Not a diagnosis."
}
```

The probability above is an illustration of the shape, not a stored prediction. A running server returns the calibrated probability from the loaded model.

`data_sufficiency` is `VERY_LOW`, `LOW`, `MODERATE`, or `HIGH` from the count of vital and laboratory values present at that hour: 0, 1–2, 3–5, and 6 or more. It is not clinical confidence. It does not change `risk_probability`.

`direction` is the sign of the summed pre-calibration tree contribution for that clinical variable. Explanations are limited to the variables with the largest absolute sums. Engineered column names are not returned.

`POST /api/predict-sequence` returns `model_name`, `model_version`, `prediction_horizon`, `disclaimer`, and `predictions`. Each prediction has the same per-hour fields as the single-observation response, plus `hour` when an hour was submitted. `risk_trend` is `increasing`, `decreasing`, `unchanged`, or null on the first scored hour. Later hours do not receive future measurements.

## Errors

Invalid JSON or a body that fails the schema returns HTTP 422. The `detail` field is either a short message or FastAPI's validation list. It does not include a stack trace.

A clinically named mistake, such as an unknown parameter or a reversed timeline, returns:

```json
{"error": "invalid_input", "detail": "observations must be in chronological order"}
```

If the model artifacts cannot be loaded, the process does not start. A failed artifact check raised in-process uses HTTP 503 and `error` of `model_unavailable`.

## What stays on the server

The browser does not receive the model file, the calibrator, the feature builder, or the threshold file. Those stay in the API process.
