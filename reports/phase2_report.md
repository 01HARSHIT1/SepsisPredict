# Phase 2 model report

This file is generated from `reports/phase2_metrics.json`. `SepsisLabel` is the PhysioNet 2019 early-warning label, equal to 1 from six hours before the challenge onset for patients who meet that definition. It is not a contemporaneous diagnosis. A one-variable result below is an experimental operating characteristic, not evidence of clinical reliability.

## 1. Feature matrix definition

Contract version `1`. Schema version `features-be7550bb5e73abe1`. 311 features in `artifacts/feature_definitions/feature_spec.json` order.

Static variables are Age, Gender, Unit1, Unit2, and HospAdmTime. They have a carried value and a missing indicator, and they have no hourly delta. ICULOS is the submitted time value only. Near-hourly variables keep raw, missing, last, time since measurement, one-hour and three-hour deltas of measured values, and backward 3-hour and 6-hour summaries. Intermittent and sparse variables keep the current raw value separate from the last observed value, the missing indicator, time since measurement, change from the previous measurement, and a 24-hour measurement count. A carried-forward lab is not treated as a fresh measurement: the raw value is missing on hours without a new measurement.

- `near_hourly` (6): HR, O2Sat, SBP, MAP, DBP, Resp. Suffixes: raw, missing, last, time_since_measurement, delta_1h, delta_3h, mean_3h, mean_6h, median_6h, min_6h, max_6h, std_6h, slope_6h.
- `intermittent` (26): Temp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Potassium, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets. Suffixes: raw, missing, last, time_since_measurement, change_from_previous_measurement, change_direction, measurements_in_24h, mean_6h.
- `sparse` (2): Phosphate, Bilirubin_total. Suffixes: raw, missing, last, time_since_measurement, change_from_previous_measurement, change_direction, measurements_in_24h.
- `static` (5): Age, Gender, Unit1, Unit2, HospAdmTime. Suffixes: value, missing.
- `time_context` (1): ICULOS. Suffixes: (raw time value).

## 2. Preprocessing pipeline

Training and inference both call `assemble_patient`.

1. Parse the request into a 40-column hourly matrix. Omitted keys and null stay missing. Numeric 0 stays 0.
2. If the request uses `hour`, sort by hour and insert missing rows for gaps. Score only the submitted hours.
3. At hour t, every feature reads measurements from hour 0 through t.
4. Logistic regression then replaces non-finite features with the training reservoir median and applies the training `StandardScaler` mean and scale. Columns with no training observations use median 0. XGBoost does not impute or scale; non-finite values remain missing.
5. The model emits a probability of `SepsisLabel`. The saved calibrator, chosen on validation, maps that probability. Explanations use the pre-calibration log-odds contributions.
6. Feature order is read from `artifacts/metadata/serving_model.json`. It is not hardcoded in the response builder.

Protocol: {
  "split_seed": 2019,
  "target": "SepsisLabel",
  "serving_selection": "validation normalized utility, then validation AUPRC",
  "calibration_selection": "lower validation Brier of Platt versus isotonic",
  "threshold_selection": "validation normalized utility after calibration for the serving model",
  "masking": "Retrain XGBoost on each variable subset. Count masks use the training-set raw observation rate, with official column order as the tie break. These runs are not used to choose the serving model.",
  "xgboost_rounds": 120,
  "logistic_epochs": 5,
  "gru": "not trained in this phase"
}

## 3. Logistic regression baseline

### logistic_unweighted

Hyperparameters: `{"loss": "log_loss", "penalty": "l2", "alpha": 0.0001, "epochs": 5, "class_weight": "none"}`

validation:

| metric | value |
| --- | --- |
| hours | 231961 |
| positive hours | 4212 |
| positive-hour rate | 0.0181582 |
| AUROC | 0.673009 |
| AUPRC | 0.0486589 |
| Brier | 0.0230083 |
| calibration slope | 0.308404 |
| calibration intercept | -2.51611 |
| threshold | 0.03 |
| sensitivity | 0.337132 |
| specificity | 0.888197 |
| precision | 0.0528215 |
| F1 | 0.091333 |
| normalized utility | 0.172061 |
| median warning lead hours | 28 |
| sepsis patients with onset inside file | 439 |
| of those, patients with an alarm | 309 |

test_at_validation_threshold:

| metric | value |
| --- | --- |
| hours | 230707 |
| positive hours | 4176 |
| positive-hour rate | 0.0181009 |
| AUROC | 0.682374 |
| AUPRC | 0.0612452 |
| Brier | 0.0221369 |
| calibration slope | 0.349584 |
| calibration intercept | -2.3523 |
| threshold | 0.03 |
| sensitivity | 0.377395 |
| specificity | 0.886956 |
| precision | 0.0579753 |
| F1 | 0.10051 |
| normalized utility | 0.206561 |
| median warning lead hours | 31 |
| sepsis patients with onset inside file | 437 |
| of those, patients with an alarm | 295 |

The threshold in `test_at_validation_threshold` was chosen on uncalibrated validation utility. It is not the serving threshold when a calibrator is applied afterward.

### logistic_balanced

Hyperparameters: `{"loss": "log_loss", "penalty": "l2", "alpha": 0.0001, "epochs": 5, "class_weight": "balanced"}`

validation:

| metric | value |
| --- | --- |
| hours | 231961 |
| positive hours | 4212 |
| positive-hour rate | 0.0181582 |
| AUROC | 0.585928 |
| AUPRC | 0.0308385 |
| Brier | 0.195812 |
| calibration slope | 0.0415115 |
| calibration intercept | -3.77618 |
| threshold | 0.99 |
| sensitivity | 0.269468 |
| specificity | 0.87268 |
| precision | 0.0376676 |
| F1 | 0.066096 |
| normalized utility | 0.0757343 |
| median warning lead hours | 33.5 |
| sepsis patients with onset inside file | 439 |
| of those, patients with an alarm | 358 |

test_at_validation_threshold:

| metric | value |
| --- | --- |
| hours | 230707 |
| positive hours | 4176 |
| positive-hour rate | 0.0181009 |
| AUROC | 0.617281 |
| AUPRC | 0.0399371 |
| Brier | 0.19565 |
| calibration slope | 0.0525512 |
| calibration intercept | -3.75213 |
| threshold | 0.99 |
| sensitivity | 0.323994 |
| specificity | 0.871519 |
| precision | 0.0444218 |
| F1 | 0.0781313 |
| normalized utility | 0.124074 |
| median warning lead hours | 34 |
| sepsis patients with onset inside file | 437 |
| of those, patients with an alarm | 344 |

The threshold in `test_at_validation_threshold` was chosen on uncalibrated validation utility. It is not the serving threshold when a calibrator is applied afterward.


## 4. Gradient-boosted model

### xgboost_unweighted

Hyperparameters: `{"objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": 4, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "tree_method": "hist", "max_bin": 128, "seed": 2019, "nthread": 4, "num_boost_round": 120, "early_stopping_rounds": 20, "scale_pos_weight": 1.0, "best_iteration": 77}`

validation:

| metric | value |
| --- | --- |
| hours | 231961 |
| positive hours | 4212 |
| positive-hour rate | 0.0181582 |
| AUROC | 0.8429 |
| AUPRC | 0.130678 |
| Brier | 0.0167089 |
| calibration slope | 1.06819 |
| calibration intercept | 0.244971 |
| threshold | 0.03 |
| sensitivity | 0.587132 |
| specificity | 0.882805 |
| precision | 0.0847963 |
| F1 | 0.14819 |
| normalized utility | 0.414399 |
| median warning lead hours | 27.5 |
| sepsis patients with onset inside file | 439 |
| of those, patients with an alarm | 354 |

test_at_validation_threshold:

| metric | value |
| --- | --- |
| hours | 230707 |
| positive hours | 4176 |
| positive-hour rate | 0.0181009 |
| AUROC | 0.839063 |
| AUPRC | 0.140502 |
| Brier | 0.0166124 |
| calibration slope | 1.06087 |
| calibration intercept | 0.234536 |
| threshold | 0.03 |
| sensitivity | 0.564655 |
| specificity | 0.889176 |
| precision | 0.085861 |
| F1 | 0.149057 |
| normalized utility | 0.397456 |
| median warning lead hours | 27 |
| sepsis patients with onset inside file | 437 |
| of those, patients with an alarm | 345 |

The threshold in `test_at_validation_threshold` was chosen on uncalibrated validation utility. It is not the serving threshold when a calibrator is applied afterward.

### xgboost_scale_pos_weight

Hyperparameters: `{"objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": 4, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "tree_method": "hist", "max_bin": 128, "seed": 2019, "nthread": 4, "num_boost_round": 120, "early_stopping_rounds": 20, "scale_pos_weight": 54.79383449405981, "best_iteration": 59}`

validation:

| metric | value |
| --- | --- |
| hours | 231961 |
| positive hours | 4212 |
| positive-hour rate | 0.0181582 |
| AUROC | 0.840412 |
| AUPRC | 0.124647 |
| Brier | 0.148767 |
| calibration slope | 1.20116 |
| calibration intercept | -3.85613 |
| threshold | 0.57 |
| sensitivity | 0.626306 |
| specificity | 0.859499 |
| precision | 0.0761613 |
| F1 | 0.135808 |
| normalized utility | 0.416294 |
| median warning lead hours | 29 |
| sepsis patients with onset inside file | 439 |
| of those, patients with an alarm | 355 |

test_at_validation_threshold:

| metric | value |
| --- | --- |
| hours | 230707 |
| positive hours | 4176 |
| positive-hour rate | 0.0181009 |
| AUROC | 0.838283 |
| AUPRC | 0.124724 |
| Brier | 0.146723 |
| calibration slope | 1.16812 |
| calibration intercept | -3.83662 |
| threshold | 0.57 |
| sensitivity | 0.594828 |
| specificity | 0.864994 |
| precision | 0.0751202 |
| F1 | 0.133394 |
| normalized utility | 0.392097 |
| median warning lead hours | 29 |
| sepsis patients with onset inside file | 437 |
| of those, patients with an alarm | 353 |

The threshold in `test_at_validation_threshold` was chosen on uncalibrated validation utility. It is not the serving threshold when a calibrator is applied afterward.


## 5. Calibration

Serving model: `xgboost_scale_pos_weight` (xgboost).

Maximum validation normalized utility of uncalibrated probabilities. Ties use validation AUPRC. The calibrator and the decision threshold are then fit on validation only.

Validation ranking used for selection:

| model | validation utility | validation AUPRC |
| --- | --- | --- |
| xgboost_scale_pos_weight | 0.416294 | 0.124647 |
| xgboost_unweighted | 0.414399 | 0.130678 |
| logistic_unweighted | 0.172061 | 0.0486589 |
| logistic_balanced | 0.0757343 | 0.0308385 |

Chosen calibrator: `isotonic`.
Validation Brier, Platt: 0.0168104.
Validation Brier, isotonic: 0.0167256.
lower validation Brier; that Brier is fit and scored on the same rows

Validation operating point after calibration (threshold selection uses these same validation rows):

Threshold `0.03`, normalized utility `0.419104`.

Test metrics of the calibrated serving model at that threshold:

| metric | value |
| --- | --- |
| hours | 230707 |
| positive hours | 4176 |
| positive-hour rate | 0.0181009 |
| AUROC | 0.837782 |
| AUPRC | 0.117987 |
| Brier | 0.0167437 |
| calibration slope | 0.943672 |
| calibration intercept | -0.181818 |
| threshold | 0.03 |
| sensitivity | 0.559626 |
| specificity | 0.883773 |
| precision | 0.0815252 |
| F1 | 0.142318 |
| normalized utility | 0.385795 |
| median warning lead hours | 30 |
| sepsis patients with onset inside file | 437 |
| of those, patients with an alarm | 341 |

## 6. Feature masking

Count masks keep the variables with the highest fraction of training hours observed. Ties follow the official 40-column order. The full-information models above are the baseline. Each mask below is a separately trained XGBoost model. Its threshold is chosen on its own validation utility and then applied to test. Mask results are not used to pick the serving model.

Training observation ranking:

| rank | variable | observed hours | observation rate |
| --- | --- | --- | --- |
| 1 | Age | 1089542 | 1 |
| 2 | Gender | 1089542 | 1 |
| 3 | ICULOS | 1089542 | 1 |
| 4 | HospAdmTime | 1089534 | 0.999993 |
| 5 | HR | 980553 | 0.899968 |
| 6 | MAP | 952215 | 0.873959 |
| 7 | O2Sat | 945733 | 0.86801 |
| 8 | SBP | 929886 | 0.853465 |
| 9 | Resp | 921374 | 0.845653 |
| 10 | DBP | 747382 | 0.68596 |
| 11 | Unit1 | 657553 | 0.603513 |
| 12 | Unit2 | 657553 | 0.603513 |
| 13 | Temp | 367773 | 0.337548 |
| 14 | Glucose | 186615 | 0.171278 |
| 15 | Potassium | 101428 | 0.0930923 |
| 16 | Hct | 96426 | 0.0885014 |
| 17 | FiO2 | 90867 | 0.0833993 |
| 18 | Hgb | 80524 | 0.0739063 |
| 19 | pH | 75604 | 0.0693906 |
| 20 | BUN | 74878 | 0.0687243 |
| 21 | WBC | 69857 | 0.0641159 |
| 22 | Magnesium | 68900 | 0.0632376 |
| 23 | Creatinine | 66555 | 0.0610853 |
| 24 | Platelets | 64862 | 0.0595314 |
| 25 | Calcium | 64295 | 0.059011 |
| 26 | PaCO2 | 60613 | 0.0556316 |
| 27 | BaseExcess | 59163 | 0.0543008 |
| 28 | Chloride | 49424 | 0.0453622 |
| 29 | HCO3 | 45567 | 0.0418222 |
| 30 | Phosphate | 43979 | 0.0403647 |
| 31 | EtCO2 | 40838 | 0.0374818 |
| 32 | SaO2 | 37528 | 0.0344438 |
| 33 | PTT | 32168 | 0.0295243 |
| 34 | Lactate | 28988 | 0.0266057 |
| 35 | AST | 17772 | 0.0163114 |
| 36 | Alkalinephos | 17585 | 0.0161398 |
| 37 | Bilirubin_total | 16300 | 0.0149604 |
| 38 | TroponinI | 10336 | 0.00948655 |
| 39 | Fibrinogen | 7340 | 0.00673678 |
| 40 | Bilirubin_direct | 2159 | 0.00198157 |

| mask | variables | features | validation AUPRC | validation utility | test AUPRC | test utility | test sensitivity | test precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_30 | 30 | 232 | 0.125342 | 0.413133 | 0.1259 | 0.391717 | 0.623324 | 0.0700352 |
| top_20 | 20 | 153 | 0.12244 | 0.39994 | 0.117881 | 0.375674 | 0.617098 | 0.066517 |
| top_15 | 15 | 113 | 0.117714 | 0.378686 | 0.110498 | 0.359025 | 0.59387 | 0.0658734 |
| top_10 | 10 | 85 | 0.105796 | 0.344081 | 0.103977 | 0.319322 | 0.567529 | 0.0605364 |
| top_5 | 5 | 20 | 0.0910562 | 0.294463 | 0.0831795 | 0.300486 | 0.422174 | 0.085334 |
| top_3 | 3 | 5 | 0.0778988 | 0.280466 | 0.0688632 | 0.278687 | 0.353927 | 0.103916 |
| top_2 | 2 | 4 | 0.0196782 | 0.000377816 | 0.0176629 | -0.0049384 | 0.00694444 | 0.0168899 |
| top_1 | 1 | 2 | 0.0201748 | 0 | 0.019454 | 0 | 0 | nan |
| vitals_only | 8 | 94 | 0.0668883 | 0.252729 | 0.0619721 | 0.227364 | 0.489703 | 0.0516114 |
| labs_only | 26 | 206 | 0.0744685 | 0.285341 | 0.0622597 | 0.265514 | 0.548132 | 0.0519613 |
| vitals_without_labs | 14 | 105 | 0.118242 | 0.383929 | 0.116882 | 0.359417 | 0.556513 | 0.0734304 |
| selected_labs_without_vitals | 5 | 39 | 0.0695181 | 0.259751 | 0.0630815 | 0.224794 | 0.444444 | 0.0535102 |
| fast_vitals_only | 6 | 78 | 0.0480865 | 0.140721 | 0.0468762 | 0.146624 | 0.413314 | 0.0430789 |
| static_and_time | 6 | 11 | 0.0810081 | 0.279455 | 0.069274 | 0.274334 | 0.341715 | 0.110895 |
| hr_only | 1 | 13 | 0.0289111 | 0.0391809 | 0.0294369 | 0.050363 | 0.260776 | 0.035179 |
| lactate_only | 1 | 8 | 0.0540579 | 0.196492 | 0.0458824 | 0.15863 | 0.379789 | 0.0464707 |

Variables in each mask:

- `top_30`: Age, Gender, ICULOS, HospAdmTime, HR, MAP, O2Sat, SBP, Resp, DBP, Unit1, Unit2, Temp, Glucose, Potassium, Hct, FiO2, Hgb, pH, BUN, WBC, Magnesium, Creatinine, Platelets, Calcium, PaCO2, BaseExcess, Chloride, HCO3, Phosphate
- `top_20`: Age, Gender, ICULOS, HospAdmTime, HR, MAP, O2Sat, SBP, Resp, DBP, Unit1, Unit2, Temp, Glucose, Potassium, Hct, FiO2, Hgb, pH, BUN
- `top_15`: Age, Gender, ICULOS, HospAdmTime, HR, MAP, O2Sat, SBP, Resp, DBP, Unit1, Unit2, Temp, Glucose, Potassium
- `top_10`: Age, Gender, ICULOS, HospAdmTime, HR, MAP, O2Sat, SBP, Resp, DBP
- `top_5`: Age, Gender, ICULOS, HospAdmTime, HR
- `top_3`: Age, Gender, ICULOS
- `top_2`: Age, Gender
- `top_1`: Age
- `vitals_only`: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
- `labs_only`: BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets
- `vitals_without_labs`: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
- `selected_labs_without_vitals`: Lactate, Creatinine, WBC, Platelets, Bilirubin_total
- `fast_vitals_only`: HR, O2Sat, SBP, MAP, DBP, Resp
- `static_and_time`: Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
- `hr_only`: HR
- `lactate_only`: Lactate

## 7. Temporal leakage tests

Hourly features are built by `src.features.assemble.assemble_patient` and `src.inference.features.build_feature_row`. Both read only rows at or before the current hour. `tests/test_leakage.py`, `tests/test_inference_contract.py`, and `tests/test_assemble_matches_scalar.py` check future-edit invariance, the absence of backward fill for static values, patient-level splits, and equality of the vectorized and scalar feature rows. Run `python -m pytest`.

## 8. Important features

`logistic_unweighted` largest absolute scaled coefficients:

- FiO2_raw: -1.44764
- ICULOS: 0.635986
- FiO2_change_from_previous_measurement: 0.453043
- FiO2_mean_6h: -0.373752
- BUN_measurements_in_24h: 0.364492
- Creatinine_measurements_in_24h: -0.350857
- Hct_measurements_in_24h: -0.348154
- Gender_value: 0.307625
- Platelets_time_since_measurement: -0.292228
- PTT_measurements_in_24h: 0.289291

`logistic_balanced` largest absolute scaled coefficients:

- ICULOS: 4.73983
- Creatinine_measurements_in_24h: -4.58967
- Hct_measurements_in_24h: -3.40903
- O2Sat_mean_6h: -3.38725
- BUN_measurements_in_24h: 3.35107
- Bilirubin_direct_change_from_previous_measurement: 3.3466
- PTT_measurements_in_24h: 3.24889
- Bilirubin_direct_last: 2.95305
- pH_time_since_measurement: -2.93604
- Temp_measurements_in_24h: -2.82021

`xgboost_unweighted` highest XGBoost gain:

- Calcium_change_direction: 1936.22
- Chloride_measurements_in_24h: 1448.54
- ICULOS: 1375.37
- FiO2_mean_6h: 799.545
- FiO2_change_direction: 669.486
- FiO2_time_since_measurement: 610.429
- Unit2_value: 443.15
- Resp_median_6h: 311.999
- Lactate_time_since_measurement: 309.952
- Lactate_change_from_previous_measurement: 291.205

`xgboost_scale_pos_weight` highest XGBoost gain:

- Calcium_change_direction: 25536.4
- ICULOS: 23479.3
- Lactate_last: 22277.2
- FiO2_mean_6h: 19339.9
- FiO2_time_since_measurement: 13182.8
- HR_median_6h: 9411.2
- Phosphate_change_from_previous_measurement: 9211.52
- Unit2_value: 8769.64
- BUN_mean_6h: 8225.74
- Alkalinephos_mean_6h: 7482.94

Serving-model sample: `mean_absolute_xgboost_pred_contribs`.

- ICULOS: 0.37464
- FiO2_time_since_measurement: 0.171236
- FiO2_mean_6h: 0.151792
- HospAdmTime_value: 0.134607
- Temp_last: 0.110085
- Unit1_value: 0.0984712
- WBC_last: 0.0965609
- BUN_last: 0.0917512
- SBP_min_6h: 0.0878631
- Alkalinephos_last: 0.0699453

## 9. Saved artifacts

Serving model name: `xgboost_scale_pos_weight`.
Feature schema version: `features-be7550bb5e73abe1`.
Training dataset version: `physionet-challenge-2019-1.0.0`.
Training timestamp: `2026-10-03T19:29:17.359260+00:00`.

- `artifacts/metadata/model_manifest.json`
- `artifacts/metadata/serving_model.json` (name, version, dataset, schema, target, horizon, timestamp, feature order, threshold)
- `artifacts/feature_definitions/feature_spec.json`
- `artifacts/models/` for the logistic and XGBoost candidates
- `artifacts/models/masks/` for the masking retrains
- `artifacts/preprocessors/` for the median/scaler or the native-missing feature order
- `artifacts/calibration/serving_calibrator.json`
- `data/processed/phase2/` float32 matrices, labels, and patient offsets
- `reports/phase2_metrics.json`

Split row counts:

| split | patients | hours | positive hours | sepsis patients |
| --- | --- | --- | --- | --- |
| train | 28235 | 1089542 | 19528 | 2053 |
| validation | 6051 | 231961 | 4212 | 440 |
| test | 6050 | 230707 | 4176 | 439 |

## 10. Inference command

From the project directory, after the artifacts above exist:

```
python -m src.inference.score --request request.json
python -m src.inference.score --model-info
```

`src.inference.service.predict` is the function a later FastAPI route should call. It loads the manifest, the feature spec, the preprocessor, the model, and the calibrator. No HTTP route is started in this phase.

## 11. Proposed FastAPI request schema

Mode A, current sparse values. Missing fields stay missing:

```json
{"HR": 118, "Temp": 38.7, "MAP": 64, "Resp": 25}
```

Mode B, explicit hours. The service sorts by `hour`, fills internal gaps with missing values, and returns one prediction per submitted hour:

```json
[
  {"hour": 1, "HR": 88, "MAP": 82},
  {"hour": 2, "HR": 93, "MAP": 78},
  {"hour": 3, "HR": 101, "MAP": 72}
]
```

The phase 1 body `{"observations": [{"values": {...}}]}` remains accepted. `SepsisLabel` is rejected.

## 12. Proposed FastAPI response schema

```json
{
  "contract_version": "1",
  "model_name": "string",
  "model_version": "string",
  "model_family": "logistic_regression | xgboost",
  "prediction_target": "SepsisLabel",
  "prediction_horizon": {"name": "official_sepsis_label", "description": "string"},
  "data_sufficiency": null,
  "disclaimer": "string",
  "predictions": [
    {
      "hour": "integer or null",
      "risk_probability": "number from the loaded model",
      "data_available": "integer count of finite inputs at that hour",
      "data_total": 40,
      "data_completeness": "data_available / 40",
      "data_sufficiency": null,
      "historical_hours": "integer length of the timeline through this hour",
      "risk_trend": "increasing | decreasing | unchanged | null",
      "explanations": [
        {
          "feature": "string from the saved feature order",
          "contribution": "number",
          "contribution_space": "log_odds_before_calibration"
        }
      ]
    }
  ]
}
```

## What this phase does not contain

No GRU, LSTM, or temporal-attention model is trained. No Next.js page and no FastAPI route is implemented. `data_sufficiency` is null. Explanations are model contributions in pre-calibration log-odds space, not prose diagnoses.
