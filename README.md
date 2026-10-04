# Early sepsis risk research prototype

Research early-warning estimates for the PhysioNet/Computing in Cardiology Challenge 2019 `SepsisLabel`. The application accepts whatever subset of the 40 hourly inputs is available and returns a calibrated model probability. It does not diagnose sepsis.

This application is a research/educational early-warning prototype based on the PhysioNet 2019 SepsisLabel. It does not diagnose sepsis and must not be used as a substitute for clinical assessment.

## Architecture

```
Next.js
  → FastAPI
  → src.inference.service.predict
  → saved flexible_xgboost
  → shared feature pipeline
  → isotonic calibration
  → estimated risk and grouped explanations
```

The browser sends raw observations only. `frontend/lib/api.ts` is the only HTTP client. Feature construction, imputation policy, calibration, and explanations stay in Python. The API loads `artifacts/metadata/phase5_model_manifest.json` once at startup and does not fall back to another model. The command-line scorer calls the same `predict` function; its default manifest is still the Phase 2 model and is not what the web app serves.

## Dataset

Public training sets A (20,336 patients) and B (20,000 patients):

https://physionet.org/content/challenge-2019/1.0.0/

DOI: 10.13026/v64v-d857

Pooled: 40,336 patients, 1,552,210 hours, 2,932 patients with at least one positive label (7.27%), 27,916 positive hours (1.80%). Patient-level splits use seed 2019: train 28,235, validation 6,051, test 6,050. Raw files stay under `data/raw/` and are not served by Next.js.

`SepsisLabel` is 1 from six hours before the challenge onset for patients who meet that definition, and 0 otherwise. It is not a contemporaneous diagnosis.

## Machine learning

Phase 1 fixes the representations later models may use. Phase 2 trains logistic regression and histogram XGBoost, then selects `xgboost_scale_pos_weight` with isotonic calibration and threshold 0.03. Phase 4 adds one missingness-aware flexible XGBoost, temporal ablations, hospital transfer, and GRU windows. Phase 5 does not train a new family. It compares `flexible_xgboost` and `temporal_last` on validation only and serves `flexible_xgboost` because sparse-input utility is part of the application score. Reports: `reports/phase1_report.md`, `reports/phase2_report.md`, `reports/phase4_report.md`, `reports/phase5_report.md`, and `reports/final_system_report.md`.

Seeds: patient splits 2019, flexible-mask training 2019, flexible-mask evaluation 2024, Phase 5 subset draws 2019. Feature schema `features-be7550bb5e73abe1`. Serving model version is stored in the Phase 5 manifest. Python packages are pinned in `requirements.txt`. Frontend packages are pinned in `frontend/package.json` and `frontend/package-lock.json`.

## Limitations

Hospital-transfer utility is lower than pooled utility. One-variable scores can order hours while staying below the research threshold, so sensitivity at 0.03 can be zero. `data_sufficiency` counts vital and laboratory values present at the scored hour. It is not clinical confidence. The 0.03 cutoff is a research early-warning threshold chosen on validation utility. A displayed percentage is the model's calibrated probability for the challenge label, not a clinical probability that a person has sepsis.

## Backend

From the repository root:

```
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Startup fails if the Phase 5 manifest, model, preprocessor, calibration artifact, metadata, or feature schema do not match.

## Frontend

```
cd frontend
npm install
npm run dev
```

The development server listens on port 3000.

## Environment

`frontend/.env.example` and `frontend/.env.local`:

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Components do not contain that host. The API reads `CORS_ORIGINS`. The default is `http://localhost:3000,http://127.0.0.1:3000`. A wildcard origin is rejected. `.env` and `frontend/.env.local` are gitignored.

## Testing

```
python -m pytest
cd frontend
npm test
npm run build
```

## API

| method | path | role |
| --- | --- | --- |
| GET | `/health` | Process status. Does not score the model. |
| GET | `/api/model-info` | Loaded model metadata. Does not score the model. |
| POST | `/api/predict` | One current observation. |
| POST | `/api/predict-sequence` | One score per submitted hour, in order. |

The request and response fields are in `docs/frontend_api_contract.md`. A sequence may contain at most 336 observations, and each observation at most 40 values. Omitted fields and JSON null stay missing. Numeric 0 is a measured zero.
