# Phase 1 dataset audit

This is a research audit of the public PhysioNet/Computing in Cardiology Challenge 2019 training data.
No sepsis risk model has been trained. No accuracy, AUROC, utility, or calibration number in this report comes from a model.
The system described here is not a medical device and does not diagnose sepsis.

Dataset: Reyna et al., PhysioNet, version 1.0.0, DOI 10.13026/v64v-d857.
Source: https://physionet.org/content/challenge-2019/1.0.0/

## A. Dataset statistics

| Set | Patients | Hours | Sepsis patients | Patient prevalence | Positive hours | Hour prevalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 20,336 | 790,215 | 1,790 | 8.80% | 17,136 | 2.17% |
| B | 20,000 | 761,995 | 1,142 | 5.71% | 10,780 | 1.41% |
| pooled | 40,336 | 1,552,210 | 2,932 | 7.27% | 27,916 | 1.80% |

Record length is the number of hourly rows in each patient file.
- Training set A: median 39.0 hours, mean 38.9, IQR 25.0 to 47.0, range 8 to 336.
- Training set B: median 38.0 hours, mean 38.1, IQR 23.0 to 47.0, range 8 to 336.
- Training set pooled: median 38.0 hours, mean 38.5, IQR 24.0 to 47.0, range 8 to 336.

ICULOS increments by 1 hour in all but 0 patients, and starts at 1 in all but 8793 patients. Row order is therefore the hourly clock for windowing, except where those counts are nonzero.
Files that failed to parse: 0.

## B. Sepsis prevalence

SepsisLabel is the challenge target. For a patient who meets the challenge sepsis definition, the label is 1 from six hours before the operationalized onset (`t_sepsis - 6`) through the rest of the file. It is 0 before that, and it is 0 for every hour of a patient who does not meet the definition. It is not a confirmation that sepsis is clinically diagnosed at that hour.

Pooled patient prevalence: 2,932 of 40,336 patients have at least one positive label (7.27%).
Pooled hour prevalence: 27,916 of 1,552,210 hours are positive (1.80%).
Of sepsis-labeled patients, 426 are already positive at the first recorded hour, so the file contains no pre-label hour. 2,506 have at least one leading negative hour.
Using the challenge utility reconstruction (first positive index + 6 hours), the reconstructed onset index falls inside the file for 2,917 sepsis-labeled patients. When it falls outside the file, the record ends during the early window.
Labels that are not monotonic non-decreasing: 0 patients. A non-monotonic label would make a single onset time ambiguous.

Set-specific prevalence is in the table above. A difference between set A and set B is a shift, not a pooled constant.

## C. Feature availability

- Bilirubin_direct (lab): present at least once in 5.1% of patients and in 0.2% of hours.
- EtCO2 (vital): present at least once in 8.0% of patients and in 3.7% of hours.
- Fibrinogen (lab): present at least once in 11.2% of patients and in 0.7% of hours.
- TroponinI (lab): present at least once in 17.5% of patients and in 1.0% of hours.
- Lactate (lab): present at least once in 31.0% of patients and in 2.7% of hours.
- SaO2 (lab): present at least once in 32.4% of patients and in 3.5% of hours.
- BaseExcess (lab): present at least once in 32.7% of patients and in 5.4% of hours.
- Alkalinephos (lab): present at least once in 35.1% of patients and in 1.6% of hours.
- Bilirubin_total (lab): present at least once in 35.3% of patients and in 1.5% of hours.
- AST (lab): present at least once in 35.6% of patients and in 1.6% of hours.
- FiO2 (lab): present at least once in 44.2% of patients and in 8.3% of hours.
- PaCO2 (lab): present at least once in 45.5% of patients and in 5.6% of hours.
- pH (lab): present at least once in 46.9% of patients and in 6.9% of hours.
- HCO3 (lab): present at least once in 50.1% of patients and in 4.2% of hours.
- PTT (lab): present at least once in 50.2% of patients and in 2.9% of hours.
- Chloride (lab): present at least once in 53.1% of patients and in 4.5% of hours.
- Unit1 (static): present at least once in 61.3% of patients and in 60.6% of hours.
- Unit2 (static): present at least once in 61.3% of patients and in 60.6% of hours.
- Phosphate (lab): present at least once in 70.2% of patients and in 4.0% of hours.
- DBP (vital): present at least once in 81.6% of patients and in 68.7% of hours.
- Calcium (lab): present at least once in 86.8% of patients and in 5.9% of hours.
- Magnesium (lab): present at least once in 87.8% of patients and in 6.3% of hours.
- WBC (lab): present at least once in 93.5% of patients and in 6.4% of hours.
- Platelets (lab): present at least once in 93.6% of patients and in 5.9% of hours.
- Hgb (lab): present at least once in 93.9% of patients and in 7.4% of hours.
- Hct (lab): present at least once in 94.3% of patients and in 8.9% of hours.
- Creatinine (lab): present at least once in 94.9% of patients and in 6.1% of hours.
- BUN (lab): present at least once in 95.0% of patients and in 6.9% of hours.
- Potassium (lab): present at least once in 95.4% of patients and in 9.3% of hours.
- Glucose (lab): present at least once in 96.1% of patients and in 17.1% of hours.
- Temp (vital): present at least once in 99.3% of patients and in 33.8% of hours.
- SBP (vital): present at least once in 99.3% of patients and in 85.4% of hours.
- MAP (vital): present at least once in 99.7% of patients and in 87.5% of hours.
- Resp (vital): present at least once in 99.8% of patients and in 84.6% of hours.
- O2Sat (vital): present at least once in 100.0% of patients and in 86.9% of hours.
- HR (vital): present at least once in 100.0% of patients and in 90.1% of hours.
- HospAdmTime (static): present at least once in 100.0% of patients and in 100.0% of hours.
- Age (static): present at least once in 100.0% of patients and in 100.0% of hours.
- Gender (static): present at least once in 100.0% of patients and in 100.0% of hours.
- ICULOS (time_context): present at least once in 100.0% of patients and in 100.0% of hours.

## D. Variable measurement frequency

A gap is the difference in row index between consecutive non-missing values. When ICULOS increments by 1 each row, that difference is hours between real measurements.

- HR: 90.1% of hours recorded, median gap 1.0 h, 90th percentile gap 1.0 h, maximum gap 28 h, 40,319 patients with a repeat measurement, of whom 99.9% show a numeric change.
- MAP: 87.5% of hours recorded, median gap 1.0 h, 90th percentile gap 1.0 h, maximum gap 45 h, 40,106 patients with a repeat measurement, of whom 100.0% show a numeric change.
- O2Sat: 86.9% of hours recorded, median gap 1.0 h, 90th percentile gap 1.0 h, maximum gap 44 h, 40,296 patients with a repeat measurement, of whom 99.0% show a numeric change.
- SBP: 85.4% of hours recorded, median gap 1.0 h, 90th percentile gap 1.0 h, maximum gap 116 h, 39,980 patients with a repeat measurement, of whom 100.0% show a numeric change.
- Resp: 84.6% of hours recorded, median gap 1.0 h, 90th percentile gap 1.0 h, maximum gap 64 h, 40,200 patients with a repeat measurement, of whom 99.3% show a numeric change.
- DBP: 68.7% of hours recorded, median gap 1.0 h, 90th percentile gap 1.0 h, maximum gap 173 h, 32,734 patients with a repeat measurement, of whom 100.0% show a numeric change.
- Temp: 33.8% of hours recorded, median gap 2.0 h, 90th percentile gap 5.0 h, maximum gap 114 h, 39,583 patients with a repeat measurement, of whom 99.7% show a numeric change.
- Glucose: 17.1% of hours recorded, median gap 3.0 h, 90th percentile gap 10.0 h, maximum gap 77 h, 33,327 patients with a repeat measurement, of whom 99.6% show a numeric change.
- Potassium: 9.3% of hours recorded, median gap 6.0 h, 90th percentile gap 23.0 h, maximum gap 72 h, 30,600 patients with a repeat measurement, of whom 94.6% show a numeric change.
- Hct: 8.9% of hours recorded, median gap 6.0 h, 90th percentile gap 23.0 h, maximum gap 96 h, 29,286 patients with a repeat measurement, of whom 97.3% show a numeric change.
- FiO2: 8.3% of hours recorded, median gap 3.0 h, 90th percentile gap 5.0 h, maximum gap 200 h, 14,569 patients with a repeat measurement, of whom 82.2% show a numeric change.
- Hgb: 7.4% of hours recorded, median gap 8.0 h, 90th percentile gap 24.0 h, maximum gap 96 h, 28,066 patients with a repeat measurement, of whom 95.3% show a numeric change.
- pH: 6.9% of hours recorded, median gap 2.0 h, 90th percentile gap 9.0 h, maximum gap 326 h, 14,488 patients with a repeat measurement, of whom 96.4% show a numeric change.
- BUN: 6.9% of hours recorded, median gap 10.0 h, 90th percentile gap 24.0 h, maximum gap 72 h, 28,397 patients with a repeat measurement, of whom 91.2% show a numeric change.
- WBC: 6.4% of hours recorded, median gap 10.0 h, 90th percentile gap 25.0 h, maximum gap 96 h, 26,970 patients with a repeat measurement, of whom 96.3% show a numeric change.
- Magnesium: 6.3% of hours recorded, median gap 9.0 h, 90th percentile gap 24.0 h, maximum gap 120 h, 25,110 patients with a repeat measurement, of whom 87.4% show a numeric change.
- Creatinine: 6.1% of hours recorded, median gap 12.0 h, 90th percentile gap 25.0 h, maximum gap 72 h, 27,273 patients with a repeat measurement, of whom 89.5% show a numeric change.
- Platelets: 5.9% of hours recorded, median gap 12.0 h, 90th percentile gap 25.0 h, maximum gap 96 h, 26,403 patients with a repeat measurement, of whom 99.1% show a numeric change.
- Calcium: 5.9% of hours recorded, median gap 10.0 h, 90th percentile gap 24.0 h, maximum gap 98 h, 23,803 patients with a repeat measurement, of whom 94.8% show a numeric change.
- PaCO2: 5.6% of hours recorded, median gap 3.0 h, 90th percentile gap 10.0 h, maximum gap 326 h, 13,705 patients with a repeat measurement, of whom 98.4% show a numeric change.
- BaseExcess: 5.4% of hours recorded, median gap 2.0 h, 90th percentile gap 9.0 h, maximum gap 196 h, 11,269 patients with a repeat measurement, of whom 92.7% show a numeric change.
- Chloride: 4.5% of hours recorded, median gap 7.0 h, 90th percentile gap 24.0 h, maximum gap 185 h, 16,770 patients with a repeat measurement, of whom 90.2% show a numeric change.
- HCO3: 4.2% of hours recorded, median gap 8.0 h, 90th percentile gap 24.0 h, maximum gap 128 h, 16,205 patients with a repeat measurement, of whom 87.1% show a numeric change.
- Phosphate: 4.0% of hours recorded, median gap 13.0 h, 90th percentile gap 25.0 h, maximum gap 186 h, 16,816 patients with a repeat measurement, of whom 96.9% show a numeric change.
- EtCO2: 3.7% of hours recorded, median gap 1.0 h, 90th percentile gap 1.0 h, maximum gap 242 h, 2,985 patients with a repeat measurement, of whom 98.8% show a numeric change.
- SaO2: 3.5% of hours recorded, median gap 2.0 h, 90th percentile gap 10.0 h, maximum gap 326 h, 9,054 patients with a repeat measurement, of whom 90.2% show a numeric change.
- PTT: 2.9% of hours recorded, median gap 8.0 h, 90th percentile gap 24.0 h, maximum gap 237 h, 10,697 patients with a repeat measurement, of whom 99.2% show a numeric change.
- Lactate: 2.7% of hours recorded, median gap 2.0 h, 90th percentile gap 12.0 h, maximum gap 302 h, 7,710 patients with a repeat measurement, of whom 89.9% show a numeric change.
- AST: 1.6% of hours recorded, median gap 10.0 h, 90th percentile gap 25.0 h, maximum gap 231 h, 5,645 patients with a repeat measurement, of whom 84.8% show a numeric change.
- Alkalinephos: 1.6% of hours recorded, median gap 10.0 h, 90th percentile gap 25.0 h, maximum gap 231 h, 5,571 patients with a repeat measurement, of whom 85.7% show a numeric change.
- Bilirubin_total: 1.5% of hours recorded, median gap 13.0 h, 90th percentile gap 26.0 h, maximum gap 232 h, 4,959 patients with a repeat measurement, of whom 88.3% show a numeric change.
- TroponinI: 1.0% of hours recorded, median gap 7.0 h, 90th percentile gap 16.0 h, maximum gap 268 h, 3,792 patients with a repeat measurement, of whom 80.9% show a numeric change.
- Fibrinogen: 0.7% of hours recorded, median gap 5.0 h, 90th percentile gap 14.0 h, maximum gap 256 h, 2,038 patients with a repeat measurement, of whom 86.5% show a numeric change.
- Bilirubin_direct: 0.2% of hours recorded, median gap 7.0 h, 90th percentile gap 25.0 h, maximum gap 204 h, 489 patients with a repeat measurement, of whom 68.3% show a numeric change.

## E. Dynamic, sparse, and static classification

A priori groups follow the project brief and the challenge table: eight vitals as rapid, 26 laboratory columns as sparse, and Age, Gender, Unit1, Unit2, and HospAdmTime as static. ICULOS is a time index. Empirical class uses the pooled record:

- near-hourly: at least 40% of hours observed and median gap of at most 2 hours
- intermittent: at least 5% of hours observed, or median gap of at most 12 hours, and not near-hourly
- sparse: otherwise, among vitals and labs
- static, time context, and target are assigned by role rather than by gap

- near_hourly: HR, O2Sat, SBP, MAP, DBP, Resp
- intermittent: Temp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Potassium, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets
- sparse: Phosphate, Bilirubin_total
- static: Age, Gender, Unit1, Unit2, HospAdmTime
- time_context: ICULOS
- target: SepsisLabel

A priori group and empirical class disagree for:
- Temp: a priori rapid vital is only intermittent (set A intermittent, set B intermittent).
- EtCO2: a priori rapid vital is only intermittent (set A sparse, set B intermittent).

Empirical class differs between set A and set B for:
- EtCO2: set A sparse, set B intermittent.
- AST: set A intermittent, set B sparse.
- Alkalinephos: set A intermittent, set B sparse.
- Bilirubin_direct: set A intermittent, set B sparse.
- Magnesium: set A intermittent, set B sparse.
- Phosphate: set A intermittent, set B sparse.
- Bilirubin_total: set A intermittent, set B sparse.

## F. Missingness patterns

Across patients, the median fraction of vital and laboratory cells with no recorded value is 80.5%. The mean number of vital or laboratory variables never observed in a stay is 11.8 of 34.

Hour-level association between missingness and SepsisLabel is reported in `reports/tables/missingness_vs_label.csv`. A higher positive-label rate when a variable is missing is not evidence that missingness causes sepsis. Later ICU hours are more often label-positive, and laboratory workflow changes with severity and with hospital.

Largest absolute differences in percent of hours missing, set A minus the comparison with set B:
- DBP: set A 48.1% missing, set B 13.9% missing.
- FiO2: set A 85.8% missing, set B 97.7% missing.
- Resp: set A 9.8% missing, set B 21.1% missing.
- BaseExcess: set A 89.6% missing, set B 99.8% missing.
- Glucose: set A 87.8% missing, set B 77.8% missing.
- pH: set A 88.5% missing, set B 97.8% missing.
- HCO3: set A 91.9% missing, set B 99.8% missing.
- Chloride: set A 91.7% missing, set B 99.4% missing.

Unit1 and Unit2 are administrative ICU identifiers (challenge dictionary: MICU and SICU). Each patient is summarized by the constant pair recorded in that file. `missing` means both fields were absent for the whole stay. This pattern can identify the hospital system:

| Set | Unit1 | Unit2 | Patients |
| --- | --- | --- | ---: |
| A | missing | missing | 9,522 |
| A | 0 | 1 | 5,470 |
| A | 1 | 0 | 5,344 |
| B | 0 | 1 | 6,982 |
| B | 1 | 0 | 6,923 |
| B | missing | missing | 6,095 |

Other patterns read from the same tables:

- EtCO2 is missing for 100.0% of hours in set A and 92.4% of hours in set B. In this release, a recorded EtCO2 value does not occur in set A.
- Age is present on every hour and is constant within each patient. Set A ranges from 18.11 to 89. Set B ranges from 14 to 100. The challenge dictionary defines 100 as age 90 or above, so the top code differs by hospital system.
- FiO2 is published with unit %. Of recorded values, 129,335 are at most 1 and 30 are above 1. The reservoir median is 0.50, the exact minimum is -50.00, and the exact maximum is 4,000. No rescaling was applied.
- Lactate reservoir median is 1.81 (exact minimum 0.20, exact maximum 31.0). The published unit is mg/dL. Values were not converted.
- Heart rate is first recorded after the first file row for 31,576 patients. Those early rows stay missing. They are not back-filled.
- HospAdmTime is constant for 40,335 of 40,336 patients. Among constant values the mean is -51.8 hours, the minimum is -5366.86, and the maximum is 23.99. The mean is pulled by a long negative tail.

## G. Recommended imputation strategy

These are hypotheses for the next validation experiment. They are not yet shown to improve prediction. No strategy replaces NaN with 0.

### Vitals

- HR, O2Sat, SBP, MAP, DBP, Resp: Keep the raw hour-t measurement and a missing indicator. Candidate backward features (previous value, 1-3 hour deltas, 3-6 hour mean/spread/slope) are worth testing because the series is both frequent and usually changing. A one-hour backward carry is a validation hypothesis only. Do not back-fill from the future and do not zero-fill.
- Temp, EtCO2: Treat as an irregular measurement. Store the current raw value separately from the last real value, with time since that measurement and a missing indicator. A 6-hour mean of real measurements may be tested. Do not forward-fill across a long gap as if the lab had been redrawn.

### Laboratory values

- BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Potassium, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets: Treat as an irregular measurement. Store the current raw value separately from the last real value, with time since that measurement and a missing indicator. A 6-hour mean of real measurements may be tested. Do not forward-fill across a long gap as if the lab had been redrawn.
- Phosphate, Bilirubin_total: Sparse measurement. Do not forward-fill as a current value. Represent last observed value, whether the current hour has a real measurement, time since that measurement, change between the last two real measurements when both exist, and the count of real measurements in a recent window. Leave last value missing until the first real measurement.

### Static context

- Age, Gender, Unit1, Unit2, HospAdmTime: Carry a constant within-patient value across hours only when that value is already present at the first hour of the stay. If the whole stay is missing, leave the value missing and add a missing indicator. Do not fill with 0. Do not compute hourly deltas.

### Time index

- ICULOS: Use ICULOS as the current hour index. Do not build physiologic deltas from it and do not impute it; it is structural.

### Target

- SepsisLabel: Do not impute and do not use as an input. SepsisLabel is the challenge target, already shifted to 6 hours before the reconstructed onset.

## H. Recommended temporal features

Candidate features are listed in `reports/tables/feature_dictionary.csv`. They have not been selected by a model. Twelve-hour and twenty-four-hour summaries are not in the default candidate list: many stays are shorter than that, and a feature that is missing for most rows should be added only if a later ablation shows a gain. Near-hourly candidates emphasize current value, missingness, and short backward windows. Sparse candidates emphasize last real value, time since that value, and change between real draws. Static candidates are the value and a missing indicator, with no hourly delta.

## I. Leakage risks

Automated checks that passed in this run:
- `observation_state_ignores_future_values`
- `backward_mean_ignores_future_values`
- `backward_mean_does_not_treat_nan_as_zero`
- `hour_delta_does_not_invent_a_value_for_a_missing_hour`
- `backward_slope_ignores_future_values`
- `time_since_measurement_is_nan_before_any_observation`

The official utility example `labels=[0,0,0,0,1,1]`, `predictions=[0,0,1,1,1,1]` returns 3.388888888889, matching the published scorer.

Risks that the feature code is written to avoid, and that remain process risks:
- A centered rolling window, or any back-fill, would use future measurements. The feature functions slice `values[:t+1]`.
- Treating NaN as 0 invents a measurement. Means and deltas drop missing hours instead.
- Carrying a static value backward from a later row would leak if that value was absent at the start of the stay. The audit counts how often the first real value is after hour 1.
- Reconstructing `t_sepsis` from the full label series is valid for a target or a utility score. It is not an input.
- Scaler, imputer, and calibration parameters must be fit on training patients only. This phase does not fit them.
- Randomly splitting hourly rows would put the same patient in train and test. The split below forbids that.
- Set A versus set B differences in missingness can identify the hospital system. A model can learn that signature and fail on the hidden third system.

## J. Patient splitting strategy

Seed 2019. Within each stratum of hospital system and patient-level sepsis label, patients are shuffled and allocated 70% / 15% / 15% by the largest-remainder method. Every hourly row of a patient follows that patient's split.

Assigned patients: train 28,235, validation 6,051, test 6,050. Overlap count: 0.
Identifiers are in `data/splits/`.

## K. Proposed model architecture

Not trained in this phase.

- Model A: logistic regression on current raw values plus missing indicators. Interpretable check of preprocessing.
- Model B: gradient-boosted trees on current values, missing indicators, last observations, time since measurement, the candidate backward features, and static context.
- Model C: GRU or LSTM on the hourly state sequence, using the same representation, including sparse-lab fields.
- Model D: temporal attention only after A-C have been fit and compared. It is not assumed to be better.

Flexible-input experiments (natural missingness, random masking, and structured masking down to one variable) belong after a model exists. Performance at one variable is not claimed.
Calibration, if used, is fit on validation patients and stored separately. It is not assumed from a raw score.
Class weights are not applied yet. Hour-level and patient-level prevalence differ, so any weighting has to be judged later by recall, precision, AUPRC, false alarms, and calibration together.

## L. Proposed prediction horizon

Primary target: the official SepsisLabel at hour t. Because the challenge already sets that label to 1 at `t_sepsis - 6`, the baseline task is an early-warning label with a six-hour head start, not a classifier of confirmed sepsis at the current minute.
Evaluation of a future model should include the official utility score, which rewards predictions from 12 hours before the reconstructed onset through 3 hours after it, and penalizes late calls, missed calls, and false alarms.
A secondary research target is defined in code but not materialized as a training matrix: at a still-negative hour t, whether the official label becomes positive within a configurable horizon. That target reads future labels by design. Features at t still may not.

## Deployment contract

No HTTP service and no frontend are part of this phase. The Python modules under `src/inference/` are the boundary a later FastAPI app will call. The frontend will send JSON and display the response. It will not reimplement feature engineering.

Modules FastAPI will import:

- `src/inference/contract.py` defines the horizon name, the empty prediction slot, and risk trend from a sequence of model probabilities.
- `src/inference/features.py` parses the request and builds the feature row with `src/features/temporal.py`.
- `src/inference/artifacts.py` reads and writes the feature definition, and records where a preprocessor, model, and calibrator will be stored.
- `src/inference/service.py` refuses to score while the manifest status is `not_trained`.
- `src/data/schema.py`, `src/labels/targets.py`, and `src/evaluation/utility.py` stay on the server for column identity, targets, and the challenge utility.

Artifacts, stored apart from each other:

- `artifacts/feature_definitions/feature_spec.json` — frozen feature names, order, and empirical class. Written in this phase.
- `artifacts/metadata/model_manifest.json` — version, family, horizon, and paths. `model_version` is null.
- `artifacts/preprocessors/` — fitted training-only state, when a later experiment fits one. Not zero-fill.
- `artifacts/models/` — one model file per version. Swapping XGBoost for a GRU replaces this file and the manifest family, not the request JSON.
- `artifacts/calibration/` — optional calibrator fit on validation patients only.

Request JSON, either one hour or several hours in order:

```
{"observations": [{"values": {"HR": 120, "Temp": 39.1, "Resp": 29, "MAP": 62}}]}
```

Omitted names and null are missing. A sent 0 is a measured 0. `SepsisLabel` is rejected. Unknown names are rejected. The same parser serves the quick form and the hourly history.

The later response will carry `risk_probability`, `data_available`, `data_total`, `data_completeness`, `data_sufficiency`, `risk_trend`, `prediction_horizon`, `model_version`, and `explanations`. Availability counts are computed from the submitted hour. `data_sufficiency` stays null until a validation study defines it. `risk_probability`, `risk_trend`, and `explanations` stay absent until a model artifact exists. This phase does not fill them.

## M. Files created

See the file list written next to this report by the phase 1 runner, and the repository tree under `src/`, `tests/`, `reports/`, and `data/splits/`.

## N. Commands

From the project root `C:\CodeData\SepsisEarlyPrediction`:

```
python -m src.data.download
python -m src.data.run_phase1
python -m pytest
```

`python -m src.data.download` reads the PhysioNet training indexes and downloads `training_setA` and `training_setB`.
`python -m src.data.run_phase1` recomputes this audit, the figures, the patient split, and this report.
`python -m pytest` runs the leakage, utility, split, and synthetic-audit tests.
