# Phase 4 research report

This file is generated from `reports/phase4_metrics.json` and the unchanged Phase 2 checkpoint. `SepsisLabel` is the PhysioNet 2019 early-warning label. These experiments do not establish clinical efficacy. Count-based variable lists are availability-ranked feature masks: variables observed on the most training hours, not a ranking by predictive contribution. Age, Gender, and ICULOS lead that list because they are present on essentially every hour.

## 1. Current baseline

Phase 2 served model `xgboost_scale_pos_weight`, isotonic calibration, 311 features, 40 inputs. The test numbers below are the original Phase 2 checkpoint values.

| metric | value |
| --- | --- |
| hours | 230707 |
| positive hours | 4176 |
| negative hours | 226531 |
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

Phase 2 validation utility used for its own selection was 0.419104. Uncalibrated test AUROC was 0.838283.

## 2. Single flexible-input model

One XGBoost model was trained on all 311 features. Each training row received one deterministic mask from seed 2019. Early stopping used unmasked validation rows. The saved threshold and calibrator were fit on those same unmasked validation predictions.

Mixture: natural 0.4, availability-weighted 0.2, uniform dropout 0.15 at p=0.5, and the remainder structured. Probability 0.5 retains 20 variables in expectation, the middle of the requested information curve. Uniform dropout is 15% of training rows. Blank variable i with probability 1 - training observation rate. Rates come from the Phase 1 hourly observation counts.

| metric | value |
| --- | --- |
| hours | 230707 |
| positive hours | 4176 |
| negative hours | 226531 |
| AUROC | 0.8341 |
| AUPRC | 0.121704 |
| Brier | 0.0167154 |
| calibration slope | 0.970432 |
| calibration intercept | -0.0865995 |
| threshold | 0.03 |
| sensitivity | 0.600814 |
| specificity | 0.865016 |
| precision | 0.0758304 |
| F1 | 0.134664 |
| normalized utility | 0.398744 |

Uncalibrated test AUROC 0.834883, AUPRC 0.132146, Brier 0.152683.

Validation calibrated threshold 0.03, validation utility 0.415082.

Runtime seconds: 561.787. Process RSS after fitting: 2.02126 GB.

## 3. Flexible-input performance curve

The single flexible-input model is evaluated on the untouched pooled test set. Availability-ranked masks keep the same variables for every patient. The operating point uses the one threshold chosen on natural validation inputs. Performance is not expected to be equal across information levels.

Figure: `reports/figures/phase4_information_curve.png`.

| variables kept | hours | positive hours | negative hours | AUROC | AUPRC | Brier | sensitivity | specificity | precision | F1 | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 230707 | 4176 | 226531 | 0.8341 | 0.121704 | 0.0167154 | 0.600814 | 0.865016 | 0.0758304 | 0.134664 | 0.398744 |
| 30 | 230707 | 4176 | 226531 | 0.823651 | 0.111351 | 0.0168328 | 0.530412 | 0.893242 | 0.0839047 | 0.14489 | 0.370601 |
| 20 | 230707 | 4176 | 226531 | 0.818485 | 0.102961 | 0.0168978 | 0.539751 | 0.882665 | 0.0781716 | 0.136565 | 0.3642 |
| 15 | 230707 | 4176 | 226531 | 0.793203 | 0.0979577 | 0.0169576 | 0.471025 | 0.913915 | 0.0916247 | 0.153408 | 0.344117 |
| 10 | 230707 | 4176 | 226531 | 0.774126 | 0.0877049 | 0.0170486 | 0.448276 | 0.907386 | 0.0819184 | 0.138523 | 0.312005 |
| 5 | 230707 | 4176 | 226531 | 0.74395 | 0.0741891 | 0.0171946 | 0.421935 | 0.913981 | 0.0829255 | 0.138609 | 0.297645 |
| 3 | 230707 | 4176 | 226531 | 0.680905 | 0.0586587 | 0.0172536 | 0.358238 | 0.939006 | 0.0976948 | 0.153522 | 0.275976 |
| 2 | 230707 | 4176 | 226531 | 0.5 | 0.0181009 | 0.0177738 | 0 | 1 | null | null | 0 |
| 1 | 230707 | 4176 | 226531 | 0.5 | 0.0181009 | 0.0177738 | 0 | 1 | null | null | 0 |

The one- and two-variable availability-ranked masks are Age, then Age and Gender. On this test set those inputs produced constant scores, so AUROC is 0.5 and the natural-validation threshold raised no alarms. The three-variable mask adds ICULOS, which is present every hour; that presence is not the same as clinical importance.

Additional test protocols on the same model, using evaluation seed 2024:

- `uniform_drop_0.5`: AUROC 0.759077, AUPRC 0.0745466, utility 0.275821, hours 230707, positive 4176, negative 226531.
- `availability_weighted`: AUROC 0.779569, AUPRC 0.0907143, utility 0.324354, hours 230707, positive 4176, negative 226531.

Structured clinical-group masks on the same model:

- `drop_vitals`: AUROC 0.797669, AUPRC 0.101264, Brier 0.0170008, sensitivity 0.574952, specificity 0.839196, precision 0.0618368, F1 0.111664, utility 0.332867.
- `drop_labs`: AUROC 0.795588, AUPRC 0.100733, Brier 0.0169378, sensitivity 0.485153, specificity 0.910851, precision 0.091175, F1 0.153502, utility 0.352778.
- `drop_selected_labs`: AUROC 0.82516, AUPRC 0.108591, Brier 0.0168298, sensitivity 0.575431, specificity 0.86872, precision 0.074762, F1 0.132331, utility 0.377338.
- `only_rapid_vitals`: AUROC 0.667244, AUPRC 0.0372006, Brier 0.0176412, sensitivity 0.257184, specificity 0.905894, precision 0.0479636, F1 0.0808491, utility 0.109652.
- `only_sparse_labs`: AUROC 0.641299, AUPRC 0.0338125, Brier 0.0177229, sensitivity 0.0830939, specificity 0.985163, precision 0.0935814, F1 0.0880264, utility 0.0623322.
- `static_and_time_only`: AUROC 0.723758, AUPRC 0.0659672, Brier 0.0172084, sensitivity 0.362069, specificity 0.936636, precision 0.0952981, F1 0.150883, utility 0.275522.

## 4. Separate-model masking comparison

Separate-model masking baseline: each Phase 2 mask was its own retrained XGBoost. Those test metrics are cited from Phase 2 and were not recomputed. Names `top_N` in that checkpoint mean availability-ranked feature masks.

| mask | variables | separate-model test AUROC | separate-model test utility | single-model test AUROC | single-model test utility |
| --- | --- | --- | --- | --- | --- |
| availability-ranked 30 | 30 | 0.833797 | 0.391717 | 0.823651 | 0.370601 |
| availability-ranked 20 | 20 | 0.826198 | 0.375674 | 0.818485 | 0.3642 |
| availability-ranked 15 | 15 | 0.802876 | 0.359025 | 0.793203 | 0.344117 |
| availability-ranked 10 | 10 | 0.787879 | 0.319322 | 0.774126 | 0.312005 |
| availability-ranked 5 | 5 | 0.741877 | 0.300486 | 0.74395 | 0.297645 |
| availability-ranked 3 | 3 | 0.689668 | 0.278687 | 0.680905 | 0.275976 |
| availability-ranked 2 | 2 | 0.491845 | -0.0049384 | 0.5 | 0 |
| availability-ranked 1 | 1 | 0.524496 | 0 | 0.5 | 0 |

## 5. Hospital A to B

Train and validate on hospital A. Test on all patients from hospital B. hospital A validation only.

External patients 20000, hours 761995.

| metric | value |
| --- | --- |
| hours | 761995 |
| positive hours | 10780 |
| negative hours | 751215 |
| AUROC | 0.785734 |
| AUPRC | 0.0613362 |
| Brier | 0.0135729 |
| calibration slope | 1.02271 |
| calibration intercept | 0.155344 |
| threshold | 0.03 |
| sensitivity | 0.482839 |
| specificity | 0.903629 |
| precision | 0.0670747 |
| F1 | 0.117787 |
| normalized utility | 0.296388 |

Source-hospital validation utility 0.421383.

## 6. Hospital B to A

Train and validate on hospital B. Test on all patients from hospital A. hospital B validation only.

External patients 20336, hours 790215.

| metric | value |
| --- | --- |
| hours | 790215 |
| positive hours | 17136 |
| negative hours | 773079 |
| AUROC | 0.757265 |
| AUPRC | 0.0800847 |
| Brier | 0.0205802 |
| calibration slope | 0.794755 |
| calibration intercept | -0.671956 |
| threshold | 0.03 |
| sensitivity | 0.485936 |
| specificity | 0.857483 |
| precision | 0.0702677 |
| F1 | 0.122781 |
| normalized utility | 0.310152 |

Source-hospital validation utility 0.395631.

## 7. Administrative and site-feature ablations

Model A is the Phase 2 full model, cited above. The other rows are retrained on the pooled patient split. Removing a variable group measures association with the label in this dataset. It does not by itself decide that the group is unusable.

| model | removed or retained | test AUROC | test AUPRC | test utility | validation utility |
| --- | --- | --- | --- | --- | --- |
| A full Phase 2 model | all 311 features | 0.837782 | 0.117987 | 0.385795 | 0.419104 |
| ablation_no_admin | drop Unit1, Unit2, HospAdmTime | 0.826284 | 0.113948 | 0.377241 | 0.396684 |
| ablation_no_iculos | drop ICULOS only | 0.819762 | 0.0888821 | 0.360919 | 0.378137 |
| ablation_physiology_age | vitals, labs, and Age | 0.815892 | 0.0897399 | 0.362105 | 0.363629 |
| ablation_physiology | vitals and labs only | 0.808469 | 0.08749 | 0.344 | 0.36205 |

## 8. Feature importance stability

Mean absolute tree contributions on up to 3,000 test rows. Phase 2 model, split by hospital, and the physiology-plus-age ablation on pooled test rows.

Features in the top 20 for both hospital A and hospital B test samples:

- Alkalinephos_last
- BUN_last
- Creatinine_last
- FiO2_mean_6h
- FiO2_time_since_measurement
- HR_last
- HR_max_6h
- HospAdmTime_value
- ICULOS
- Lactate_last
- Lactate_time_since_measurement
- MAP_min_6h
- Resp_mean_6h
- SBP_min_6h
- Temp_last
- Temp_mean_6h
- Temp_measurements_in_24h
- Unit1_value
- WBC_last

Largest absolute rank changes between hospital A and hospital B:

- O2Sat_missing: A rank 307, B rank 165, change 142
- O2Sat_last: A rank 306, B rank 166, change 140
- SBP_missing: A rank 162, B rank 302, change 140
- WBC_missing: A rank 164, B rank 303, change 139
- Temp_change_direction: A rank 163, B rank 300, change 137
- Temp_missing: A rank 168, B rank 304, change 136
- SBP_raw: A rank 167, B rank 301, change 134
- HR_time_since_measurement: A rank 179, B rank 308, change 129
- Platelets_change_direction: A rank 303, B rank 174, change 129
- O2Sat_delta_1h: A rank 305, B rank 176, change 129

Administrative or time features in the compared lists:

- HospAdmTime_missing: A rank 286, B rank 309
- Unit2_missing: A rank 166, B rank 183
- Unit2_value: A rank 26, B rank 37
- Unit1_value: A rank 5, B rank 8
- Unit1_missing: A rank 183, B rank 182
- HospAdmTime_value: A rank 4, B rank 3
- ICULOS: A rank 1, B rank 1

Highest contributions, physiology-plus-age ablation, pooled test:

- FiO2_mean_6h: 0.169182
- FiO2_time_since_measurement: 0.164229
- Phosphate_change_from_previous_measurement: 0.144345
- BUN_last: 0.140343
- Temp_last: 0.111141
- Alkalinephos_last: 0.103078
- Resp_mean_6h: 0.0988851
- WBC_last: 0.0823082

## 9. False alarm analysis

### phase2_served_model

Threshold 0.03. Non-septic patients 5611, hours 205998.
False-positive hours 15715 (2.80075 per non-septic patient, 1.83089 per non-septic patient-day).
Fraction of non-septic patients ever alarmed: 0.24345.
False-alarm episodes 2557, mean duration 6.14587 hours.
Persistent episodes (at least 3 hours): 1617 (0.632382 of episodes).

### flexible_xgboost

Threshold 0.03. Non-septic patients 5611, hours 205998.
False-positive hours 19235 (3.42809 per non-septic patient, 2.24099 per non-septic patient-day).
Fraction of non-septic patients ever alarmed: 0.291035.
False-alarm episodes 3178, mean duration 6.05255 hours.
Persistent episodes (at least 3 hours): 1915 (0.60258 of episodes).


## 10. Warning lead-time analysis

Lead time is reconstructed onset minus the first alarm. A long lead is reported as a bin, not as a better result. Figure: `reports/figures/phase4_lead_time.png`.

### phase2_served_model

Onset inside the file: 437. Alarmed: 341.
Median 30, mean 47.4223, P25 8, P75 68, minimum -2, maximum 267.
More than 24 hours before onset: 0.409611.
12 to 24 hours before onset: 0.139588.
6 to 12 hours before onset: 0.0434783.
Inside the official six-hour window: 0.16476.
After reconstructed onset: 0.0228833.
Never alarmed: 0.21968.
Among alarmed patients, fraction whose first alarm is before the official positive label: 0.759531.
Among alarmed patients, fraction still alarming at the last observed hour: 0.774194.

### flexible_xgboost

Onset inside the file: 437. Alarmed: 359.
Median 27, mean 47.3816, P25 6, P75 70, minimum -2, maximum 262.
More than 24 hours before onset: 0.423341.
12 to 24 hours before onset: 0.1373.
6 to 12 hours before onset: 0.0503432.
Inside the official six-hour window: 0.183066.
After reconstructed onset: 0.02746.
Never alarmed: 0.17849.
Among alarmed patients, fraction whose first alarm is before the official positive label: 0.743733.
Among alarmed patients, fraction still alarming at the last observed hour: 0.793872.


## 11. Threshold analysis

Maximum validation normalized utility. Ties keep the lower threshold. Test rows are not used.

Validation rows only. Brier does not depend on the threshold; it is in the calibration section.

`phase2_served_model` validation utility is highest at threshold 0.03 (utility 0.419104, sensitivity 0.596391, precision 0.0842557, non-septic patients ever alarmed 0.244876).

`flexible_xgboost` validation utility is highest at threshold 0.03 (utility 0.415082, sensitivity 0.623219, precision 0.0759373, non-septic patients ever alarmed 0.294956).

The full 50-point validation tables are in `reports/phase4_metrics.json`.

## 12. Calibration

Calibrators are fit on validation predictions. The plotted curves use the untouched test set. Figure: `reports/figures/phase4_calibration_curve.png`.

Phase 2 chosen calibrator `isotonic`. Validation Brier Platt 0.0168104, isotonic 0.0167256. Test Brier 0.0167437.

Flexible model chosen calibrator `isotonic`. Validation Brier Platt 0.0168409, isotonic 0.0167918. Uncalibrated test Brier 0.152683. Calibrated test Brier 0.0167154.

Flexible-model calibrated test reliability:

| bin | n | mean predicted | fraction positive |
| --- | --- | --- | --- |
| 0 | 2985 | 1.44234e-07 | 0 |
| 1 | 40907 | 0.001163 | 0.00202899 |
| 2 | 8639 | 0.00152107 | 0.00104179 |
| 3 | 36026 | 0.00352842 | 0.00272026 |
| 4 | 25619 | 0.00516318 | 0.00538663 |
| 5 | 10877 | 0.00638192 | 0.00680335 |
| 6 | 27202 | 0.0110753 | 0.0099625 |
| 7 | 31249 | 0.0173639 | 0.0191366 |
| 8 | 19979 | 0.0264865 | 0.0310326 |
| 9 | 27224 | 0.0876256 | 0.0839333 |

## 13. Temporal feature ablation

Tree models retrained on nested feature groups. D is the Phase 2 full feature set, cited rather than retrained.

| representation | features | test AUROC | test AUPRC | test utility | validation utility |
| --- | --- | --- | --- | --- | --- |
| A current values | 40 | 0.788269 | 0.103899 | 0.345891 | 0.357999 |
| B current plus missingness | 79 | 0.792583 | 0.109455 | 0.349085 | 0.361378 |
| C plus last sparse value and time since measurement | 135 | 0.837347 | 0.115563 | 0.400773 | 0.425821 |
| D full temporal feature set, Phase 2 | 311 | 0.837782 | 0.117987 | 0.385795 | 0.419104 |

## 14. GRU results

Logistic metrics are cited from the Phase 2 checkpoint and were not refit. The logistic threshold was chosen on uncalibrated validation utility.

Logistic unweighted test: AUROC 0.682374, AUPRC 0.0612452, Brier 0.0221369, sensitivity 0.377395, specificity 0.886956, precision 0.0579753, F1 0.10051, utility 0.206561, calibration slope 0.349584.

Each GRU window is left-padded for short stays, so hours are not dropped. Coverage counts how many training hours lack a full window. Measured values are scaled with the training mean and standard deviation and then filled with 0 only beside a missingness indicator. That fill is not a fresh measurement.

| window | hours padded | fraction padded | test AUROC | test AUPRC | test Brier | test utility | validation utility | runtime seconds | process RSS GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 141175 | 0.129573 | 0.820283 | 0.103676 | 0.0168895 | 0.386245 | 0.3838 | 990.478 | 2.27283 |
| 12 | 309417 | 0.283988 | 0.811453 | 0.109006 | 0.0168539 | 0.364033 | 0.388924 | 1634 | 2.25932 |
| 24 | 613844 | 0.563396 | 0.822073 | 0.113185 | 0.0168081 | 0.376054 | 0.389448 | 2391.31 | 2.06453 |

GRU window retained for the candidate comparison: `gru_window_24`.

## 15. One-variable experiments

Single flexible-input model, one retained variable, untouched test set, natural missingness inside that variable. These are not clinical-reliability claims. `data_sufficiency` stays null. Sensitivity of 0 means no test hour reached the threshold chosen on full natural validation inputs. AUROC can still sit above 0.5 when the scores order hours but remain below that threshold.

| variable | hours | positive | negative | AUROC | AUPRC | Brier | sensitivity | specificity | precision | F1 | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HR | 230707 | 4176 | 226531 | 0.616393 | 0.0264752 | 0.0177284 | 0.182711 | 0.9049 | 0.034206 | 0.057624 | 0.0305937 |
| Temp | 230707 | 4176 | 226531 | 0.59122 | 0.0321775 | 0.0176846 | 0.197318 | 0.940803 | 0.0578896 | 0.0895166 | 0.102638 |
| MAP | 230707 | 4176 | 226531 | 0.544236 | 0.0206872 | 0.0177689 | 0 | 1 | null | null | 0 |
| Resp | 230707 | 4176 | 226531 | 0.615109 | 0.0295925 | 0.0176997 | 0.196839 | 0.927763 | 0.0478296 | 0.0769591 | 0.0825565 |
| O2Sat | 230707 | 4176 | 226531 | 0.534796 | 0.0207643 | 0.0177616 | 0 | 1 | null | null | 0 |
| SBP | 230707 | 4176 | 226531 | 0.558552 | 0.021476 | 0.0177618 | 0 | 1 | null | null | 0 |
| DBP | 230707 | 4176 | 226531 | 0.5 | 0.0181009 | 0.0177738 | 0 | 1 | null | null | 0 |
| Lactate | 230707 | 4176 | 226531 | 0.572751 | 0.0251803 | 0.0177453 | 0 | 1 | null | null | 0 |
| Creatinine | 230707 | 4176 | 226531 | 0.551904 | 0.0220052 | 0.0177619 | 0 | 1 | null | null | 0 |
| WBC | 230707 | 4176 | 226531 | 0.545349 | 0.024182 | 0.0177507 | 0 | 1 | null | null | 0 |
| Platelets | 230707 | 4176 | 226531 | 0.5 | 0.0181009 | 0.0177738 | 0 | 1 | null | null | 0 |
| Bilirubin_total | 230707 | 4176 | 226531 | 0.598219 | 0.0285031 | 0.0177469 | 0 | 1 | null | null | 0 |
| Glucose | 230707 | 4176 | 226531 | 0.5 | 0.0181009 | 0.0177738 | 0 | 1 | null | null | 0 |
| Age | 230707 | 4176 | 226531 | 0.5 | 0.0181009 | 0.0177738 | 0 | 1 | null | null | 0 |
| Gender | 230707 | 4176 | 226531 | 0.5 | 0.0181009 | 0.0177738 | 0 | 1 | null | null | 0 |
| ICULOS | 230707 | 4176 | 226531 | 0.680905 | 0.0586587 | 0.0172536 | 0.358238 | 0.939006 | 0.0976948 | 0.153522 | 0.275976 |

## 16. Provisional research serving candidate

Provisional research serving candidate: `temporal_last` (xgboost).

Highest pooled-validation normalized utility after calibration. Ties use validation AUPRC. Hospital-transfer models are reported separately and are not in this comparison because their validation population is different.

Further external and clinical validation is still required.

The single flexible-input model is `flexible_xgboost`. It is not this provisional candidate: its pooled validation utility is lower than `temporal_last` and the Phase 2 model. It remains saved under `artifacts/models/phase4_flexible_xgboost/`.

| candidate | family | flexible input | validation utility | validation AUPRC |
| --- | --- | --- | --- | --- |
| temporal_last | xgboost | False | 0.425821 | 0.122732 |
| xgboost_scale_pos_weight | xgboost | False | 0.419104 | 0.124647 |
| flexible_xgboost | xgboost | True | 0.415082 | 0.115653 |
| ablation_no_admin | xgboost | False | 0.396684 | 0.123126 |
| gru_window_24 | gru | False | 0.389448 | 0.10571 |
| ablation_no_iculos | xgboost | False | 0.378137 | 0.101093 |
| ablation_physiology_age | xgboost | False | 0.363629 | 0.0946231 |
| ablation_physiology | xgboost | False | 0.36205 | 0.0903255 |
| temporal_missing | xgboost | False | 0.361378 | 0.112244 |
| temporal_current | xgboost | False | 0.357999 | 0.1113 |

## Inference

Phase 2 remains the default `python -m src.inference.score --request request.json` manifest.

The provisional candidate is loaded with:

```
python -m src.inference.score --manifest artifacts/metadata/phase4_model_manifest.json --request request.json
```

No FastAPI route and no frontend were added. GRU windows are left-padded when a stay is shorter than the window, so those hours are still scored and are not replaced with future measurements.
