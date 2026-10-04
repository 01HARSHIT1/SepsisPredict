# Final system report

This document describes the research prototype as it stands after the end-to-end audit. Numbers below are copied from `reports/phase1_report.md`, `reports/phase2_report.md`, `reports/phase4_report.md`, and `reports/phase5_report.md`, or from the audit commands in section 17. No new model was trained for this report.

## 1. Project objective

Estimate the PhysioNet 2019 `SepsisLabel` from whatever subset of the 40 hourly inputs a user submits. The served model is the provisional research candidate `flexible_xgboost`. The web application displays that model's calibrated probability. It is a research early-warning prototype, not a diagnostic device.

## 2. Dataset

Source: PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0, DOI 10.13026/v64v-d857.

| set | patients | hours | sepsis patients | patient prevalence | positive hours | hour prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| A | 20,336 | 790,215 | 1,790 | 8.80% | 17,136 | 2.17% |
| pooled | 40,336 | 1,552,210 | 2,932 | 7.27% | 27,916 | 1.80% |

Set B is the remainder: 20,000 patients. Patient-level splits, seed 2019, stratified by hospital system and patient-level label: train 28,235, validation 6,051, test 6,050. Overlap count 0. The Phase 2 test partition has 230,707 hours and 4,176 positive hours.

## 3. Clinical problem framing

`SepsisLabel` is 1 from `t_sepsis - 6` onward for a patient who meets the challenge definition, and 0 otherwise. The modeling target is that early-warning label. It is not a confirmation that sepsis is clinically diagnosed at the scored hour. A displayed percentage is the model's calibrated probability for that label given the submitted values. It is not a clinical probability that a person has sepsis.

## 4. EDA

The Phase 1 audit recorded irregular hourly measurement, a large gap between patient prevalence (7.27%) and hour prevalence (1.80%), and very sparse laboratory streams. Age, Gender, and ICULOS are present far more often than labs such as lactate and total bilirubin. Availability is not the same thing as predictive value. Tables and figures are in `reports/phase1_report.md`, `reports/tables/`, and `reports/figures/`.

## 5. Feature engineering

One feature specification, schema `features-be7550bb5e73abe1`, builds the 311 columns used by `flexible_xgboost`. The same builder serves training and the API. The browser does not construct those columns. Current values, missingness indicators, last sparse values, time since measurement, and short temporal summaries are defined in `artifacts/feature_definitions/feature_spec.json` and assembled in Python.

## 6. Missingness strategy

Missing is not zero. A typed 0 is a measured zero. An omitted field or JSON null stays missing. Histogram XGBoost keeps native missing values. Sparse laboratory streams keep raw, missing, last, and time-since columns. They are not forward-filled and then treated as newly observed. The flexible model was trained with a deterministic mask mixture, seed 2019: 40% natural rows, 20% availability-weighted, 15% uniform dropout at probability 0.5, and 25% structured masks. Evaluation masks that are not that training mixture use seed 2024.

## 7. Temporal strategy

The label already starts six hours before the challenge onset, so the baseline task includes that head start. Tree features add last values, time since measurement, and short changes for intermittent signals. A separate temporal ablation on nested feature groups is reported in Phase 4. GRU windows of 6, 12, and 24 hours were trained and did not become the serving model. Window 24 had test AUROC 0.822073, test utility 0.376054, and validation utility 0.389448.

## 8. Model development

Phase 2 compared logistic regression and histogram XGBoost. The Phase 2 serving model is `xgboost_scale_pos_weight`, isotonic calibration, threshold 0.03. Its calibrated test metrics are AUROC 0.837782, AUPRC 0.117987, Brier 0.0167437, sensitivity 0.559626, specificity 0.883773, precision 0.0815252, and normalized utility 0.385795. Median warning lead on that test evaluation was 30 hours, with an alarm in 341 of 437 onset-in-file sepsis patients. That model remains the default command-line manifest. The API does not serve it.

Phase 4 trained one flexible-input XGBoost (`best_iteration` 69, 311 features), temporal variants, hospital-transfer models, ablations, and the GRU windows. On full natural inputs, `temporal_last` validation utility was 0.425821 and `flexible_xgboost` validation utility was 0.415082.

## 9. Flexible-input experiments

Random subsets of the 40 inputs used seed 2019, six subsets at each size except a single subset at 40. For `flexible_xgboost` on validation, mean utility was 0.133335 at k=5 and 0.205154 at k=10. For `temporal_last`, those means were 0.0155083 and 0.048852. User scenarios A through G on the flexible model are in `reports/phase5_report.md`. On the untouched test split, the flexible model utility was 0.0305937 for heart rate alone, 0.0936402 for heart rate plus respiratory rate, 0.162625 after adding temperature, and 0.370601 for the 30 most available variables.

One-variable operating points use the full-input threshold 0.03. Several variables have test AUROC above 0.5 and sensitivity 0 at that threshold. Lactate and total bilirubin were flagged sparse on the training observation rate.

## 10. Model selection

The Phase 5 rule was locked before the comparison: application score = 0.5 × pooled validation utility on natural inputs + 0.5 × mean validation utility of random subsets of size 5 and 10. Ties would use validation AUPRC, then AUROC. Test rows were not used.

| model | validation utility | validation AUPRC | validation AUROC | sparse utility | application score |
| --- | --- | --- | --- | --- | --- |
| flexible_xgboost | 0.415082 | 0.11268 | 0.837828 | 0.169244 | 0.292163 |
| temporal_last | 0.425821 | 0.11978 | 0.845345 | 0.0321802 | 0.229001 |

The provisional research serving candidate is `flexible_xgboost`. After that choice, its test metrics were AUROC 0.8341, AUPRC 0.121704, and utility 0.398744. `temporal_last` test utility was 0.400773. The API manifest is `artifacts/metadata/phase5_model_manifest.json`. Model version: `phase5-flexible_xgboost-2026-10-04T00:30:08.539603+00:00`. Threshold remains 0.03. Phase 2 and Phase 4 artifacts were not overwritten. Phase 4's earlier candidate note, which named `temporal_last` from full-input validation utility alone, is superseded by this selection rule.

## 11. Calibration

Isotonic regression was chosen for `flexible_xgboost` because its validation Brier, 0.0167918, was lower than Platt, 0.0168409. The calibrator is `artifacts/calibration/phase4_flexible_xgboost.json`. It is loaded at startup. The API does not start if that file is missing or fails to load. The displayed probability is the calibrated value. The frontend formats it to one decimal percent and does not recalibrate it.

## 12. Cross-hospital results

Transfer models train on one hospital and test on the other. They are not the serving candidate.

| experiment | external patients | hours | AUROC | AUPRC | utility | source validation utility |
| --- | --- | --- | --- | --- | --- | --- |
| Train A, test B | 20,000 | 761,995 | 0.785734 | 0.0613362 | 0.296388 | 0.421383 |
| Train B, test A | 20,336 | 790,215 | 0.757265 | 0.0800847 | 0.310152 | 0.395631 |

On the Phase 4 false-alarm count for `flexible_xgboost` at threshold 0.03, 0.291035 of non-septic patients in that evaluation were ever alarmed. A pooled test number that mixes both hospitals is not an external-hospital result.

## 13. FastAPI architecture

`PredictionService.load` reads the Phase 5 manifest once during application startup. It checks that the model file, preprocessor, calibration artifact, metadata, feature schema hash, model name, and model version agree. A missing or mismatched artifact raises `ArtifactError` and the process does not silently switch models. Routes call `src.inference.service.predict` with that loaded bundle.

`GET /health` and `GET /api/model-info` do not score the model. `POST /api/predict` requires one observation. `POST /api/predict-sequence` scores the submitted list in one call. Responses omit filesystem paths. `CORS_ORIGINS` lists explicit browser origins. The default is `http://localhost:3000,http://127.0.0.1:3000`. A wildcard is rejected. Each observation accepts at most 40 values, and a sequence accepts at most 336 observations. Unknown names, `SepsisLabel`, non-numeric values, duplicate hours, and hours that are not chronological return HTTP 422. Stack traces are not placed in those bodies.

## 14. Next.js architecture

`frontend/` is a Next.js App Router application. `frontend/lib/types.ts` matches the fields returned by `/api/predict`, `/api/predict-sequence`, and `/api/model-info`. `frontend/lib/api.ts` performs every HTTP call. Pages and components do not contain a host name. There is no second API client, no frontend feature builder, and no stored fallback probability.

The screen has a current-estimate mode and an hourly-history mode. Empty fields are omitted. A typed 0 is sent. History keeps the order shown on the page. The result card, the trend chart, and the history table render the returned probabilities. The chart also has a text list of those same percentages. Wide screens use two columns. Narrow screens stack, and the history table scrolls inside its own region.

## 15. API contract

`docs/frontend_api_contract.md` is the client contract. The probability examples in that file are shape illustrations. A running server returns the loaded model's calibrated probability.

`data_sufficiency` counts finite vital and laboratory values at the scored hour: 0 is `VERY_LOW`, 1–2 is `LOW`, 3–5 is `MODERATE`, and 6 or more is `HIGH`. Static fields, `ICULOS`, history length, and the probability do not change the band. The interface calls this Data Sufficiency and states that it indicates how much information was available to the model and is not clinical confidence.

Explanations are summed pre-calibration tree contributions grouped to clinical variables by the Python scorer. The interface shows those names as higher or lower model contribution and states that they are statistical contributions, not clinical causes.

## 16. Frontend features

Current estimate, hourly history with add and remove, service status, model information from the API, loading text without a fake progress percent, and human-readable errors for an unreachable service, HTTP 400, 422, 500, a malformed body, and a timeout. The disclaimer shown on the page is:

This application is a research/educational early-warning prototype based on the PhysioNet 2019 SepsisLabel. It does not diagnose sepsis and must not be used as a substitute for clinical assessment.

The API disclaimer is shown with it when model info or a prediction returns one. Scores at or above the research threshold are described as higher estimated model risk. Scores below it are described as lower estimated model risk. The 0.03 line is labeled a research early-warning threshold.

## 17. End-to-end test results

`python -m pytest` passed 45 tests. `npm test` in `frontend/` passed 12 tests. Those frontend tests include two calls through the HTTP client to the running API.

The audit then loaded the Phase 5 manifest through `PredictionService` and scored the same bodies the frontend sends.

Current observation, HR 118, Temp 38.7, MAP 64, Resp 25:

| field | returned value |
| --- | --- |
| risk_probability | 0.11111111111111112 |
| display format | 11.1% |
| research_early_warning | true |
| data_available | 4 / 40 |
| data_completeness | 0.1 |
| data_sufficiency | MODERATE |
| historical_hours | 1 |
| model | flexible_xgboost |

The largest grouped contributions on that call were Temp, HR, and Resp, each `increases_risk_score`. HospAdmTime, ICULOS, and SBP also appeared because missingness columns still enter the trees. Those are model contributions, not causes.

Hourly history, one sequence request:

| hour | risk_probability | display format | trend | available | sufficiency | historical hours |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0164827357552219 | 1.6% | null | 3 / 40 | MODERATE | 1 |
| 2 | 0.018867924528301886 | 1.9% | increasing | 3 / 40 | MODERATE | 2 |
| 3 | 0.039779874213836476 | 4.0% | increasing | 3 / 40 | MODERATE | 3 |
| 4 | 0.10534509600415153 | 10.5% | increasing | 3 / 40 | MODERATE | 4 |

Hour 1 was HR 88, MAP 82, Temp 37.1. Hour 2 was HR 94, MAP 78, Temp 37.4. Hour 3 was HR 105, MAP 71, Temp 38.0. Hour 4 was HR 116, MAP 64, Temp 38.6. The chart and table use these returned probabilities. They do not add or smooth points. `GET /api/model-info` for this load reported isotonic calibration, threshold 0.03, schema `features-be7550bb5e73abe1`, dataset `physionet-challenge-2019-1.0.0`, 40 supported inputs, and no filesystem path.

`npm run build` with Next.js 15.5.27 compiled, typechecked, and prerendered `/`. ESLint is not a project dependency, so the production build does not run a separate lint pass. The production server returned HTTP 200 for `/`, with the title Sepsis Early-Warning System. Submitting HR 118, Temp 38.7, MAP 64, and Resp 25 on that page displayed 11.1%, the same probability the API returned. At 390px and 768px the main column is a single track and page overflow is 0. At 1280px the input and result are side by side and page overflow is 0.

Do not run `npm run dev` and `npm run build` against the same `.next` directory at the same time. That mix produced a missing-module error until the directory was rebuilt.

## 18. Limitations

Hospital shift remains. Random subsets are a sample of combinations, not every combination. One-variable sensitivity at the research threshold can be zero while AUROC stays above 0.5. Data sufficiency has not been validated as a measure of clinical confidence. Explanations can assign contribution to a variable the user did not type, because missingness columns still enter the tree. There is no authentication, no prospective cohort, and no external health system in this prototype. Further clinical validation is still required.

## 19. Commands

Backend, from the repository root:

```
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```
cd frontend
npm install
npm run dev
```

Tests:

```
python -m pytest
cd frontend
npm test
```

Production build, from `frontend/`:

```
npm run build
```

Environment:

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Optional API setting:

```
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## 20. Medical and research disclaimer

This application is a research/educational early-warning prototype based on the PhysioNet 2019 SepsisLabel. It does not diagnose sepsis and must not be used as a substitute for clinical assessment. `research_early_warning` means the calibrated probability is at least the validation threshold stored with the model. That flag is not a diagnosis.
