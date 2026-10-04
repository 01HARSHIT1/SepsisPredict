# Phase 5 research report

This is a research early-warning model based on the PhysioNet 2019 SepsisLabel. It is not a clinical diagnosis system. No new model family was trained. The serving choice uses validation rows only. Test rows are reported after that choice.

## 1. Final candidate comparison

application_score = 0.5 * pooled validation utility on natural inputs + 0.5 * mean validation utility of random subsets of size 5 and 10. Ties use validation AUPRC, then validation AUROC. Test rows are not used.

| model | validation utility | validation AUPRC | validation AUROC | sparse utility k=5 and k=10 | application score |
| --- | --- | --- | --- | --- | --- |
| flexible_xgboost | 0.415082 | 0.11268 | 0.837828 | 0.169244 | 0.292163 |
| temporal_last | 0.425821 | 0.11978 | 0.845345 | 0.0321802 | 0.229001 |

### Validation hospital slice of the pooled models

These rows score the pooled model on validation patients from one hospital. They are not a retrained transfer experiment.

| model | hospital | hours | AUROC | AUPRC | utility |
| --- | --- | --- | --- | --- | --- |
| temporal_last | A | 117758 | 0.822834 | 0.122556 | 0.4366 |
| temporal_last | B | 114203 | 0.868593 | 0.116024 | 0.408395 |
| flexible_xgboost | A | 117758 | 0.819736 | 0.117074 | 0.437711 |
| flexible_xgboost | B | 114203 | 0.854125 | 0.106053 | 0.378498 |

### Test evaluation after the validation choice

| model | test AUROC | test AUPRC | test utility | flexible input | feature count |
| --- | --- | --- | --- | --- | --- |
| temporal_last | 0.837347 | 0.115563 | 0.400773 | False | 135 |
| flexible_xgboost | 0.8341 | 0.121704 | 0.398744 | True | 311 |

Both candidates are histogram XGBoost models with isotonic calibration and the same canonical feature builder. `temporal_last` scores 135 columns. `flexible_xgboost` scores 311 columns. Explanations for both are summed pre-calibration tree contributions grouped by clinical variable. The flexible model was trained with a deterministic mixture of natural and artificial missingness. `temporal_last` was not.

## 2. One-variable results

Each row keeps one variable and its natural missingness inside the single flexible-input model. The threshold is that model's natural-input validation threshold. A positive result here is not a claim that the variable can identify sepsis.

### Validation

| variable | training observed | split observed | sparse | AUROC | AUPRC | sensitivity | specificity | precision | F1 | Brier | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HR | 0.899968 | 0.904234 | False | 0.600681 | 0.0256545 | 0.173314 | 0.908988 | 0.0340199 | 0.0568757 | 0.0177889 | 0.0297557 |
| O2Sat | 0.86801 | 0.87169 | False | 0.526101 | 0.0200375 | 0 | 1 |  |  | 0.0178203 | 0 |
| Temp | 0.337548 | 0.341057 | False | 0.595913 | 0.0352135 | 0.221985 | 0.941387 | 0.0654579 | 0.101103 | 0.0177242 | 0.128697 |
| SBP | 0.853465 | 0.855601 | False | 0.56534 | 0.0217865 | 0 | 1 |  |  | 0.0178166 | 0 |
| MAP | 0.873959 | 0.878859 | False | 0.562229 | 0.0218475 | 0 | 1 |  |  | 0.0178218 | 0 |
| DBP | 0.68596 | 0.687137 | False | 0.5 | 0.0181582 | 0 | 1 |  |  | 0.017829 | 0 |
| Resp | 0.845653 | 0.846 | False | 0.585034 | 0.0247017 | 0.139364 | 0.931324 | 0.036172 | 0.0574364 | 0.0177925 | 0.0340967 |
| Lactate | 0.0266057 | 0.0271425 | True | 0.573338 | 0.0263429 | 0 | 1 |  |  | 0.0177979 | 0 |
| Creatinine | 0.0610853 | 0.0603032 | False | 0.561051 | 0.0227147 | 0 | 1 |  |  | 0.0178155 | 0 |
| WBC | 0.0641159 | 0.0637693 | False | 0.548848 | 0.0220531 | 0 | 1 |  |  | 0.0178146 | 0 |
| Platelets | 0.0595314 | 0.0588633 | False | 0.5 | 0.0181582 | 0 | 1 |  |  | 0.017829 | 0 |
| Bilirubin_total | 0.0149604 | 0.0148042 | True | 0.596492 | 0.0314628 | 0 | 1 |  |  | 0.0177977 | 0 |
| Glucose | 0.171278 | 0.171693 | False | 0.5 | 0.0181582 | 0 | 1 |  |  | 0.017829 | 0 |
| pH | 0.0693906 | 0.0685546 | False | 0.609356 | 0.0274316 | 0 | 1 |  |  | 0.0178003 | 0 |
| BUN | 0.0687243 | 0.0681796 | False | 0.547792 | 0.0217631 | 0 | 1 |  |  | 0.0178102 | 0 |
| Hgb | 0.0739063 | 0.0737753 | False | 0.5 | 0.0181582 | 0 | 1 |  |  | 0.017829 | 0 |
| Hct | 0.0885014 | 0.0884114 | False | 0.5 | 0.0181582 | 0 | 1 |  |  | 0.017829 | 0 |

### Test

| variable | training observed | split observed | sparse | AUROC | AUPRC | sensitivity | specificity | precision | F1 | Brier | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HR | 0.899968 | 0.903791 | False | 0.616393 | 0.0264752 | 0.182711 | 0.9049 | 0.034206 | 0.057624 | 0.0177284 | 0.0305937 |
| O2Sat | 0.86801 | 0.873589 | False | 0.534796 | 0.0207643 | 0 | 1 |  |  | 0.0177616 | 0 |
| Temp | 0.337548 | 0.339569 | False | 0.59122 | 0.0321775 | 0.197318 | 0.940803 | 0.0578896 | 0.0895166 | 0.0176846 | 0.102638 |
| SBP | 0.853465 | 0.856467 | False | 0.558552 | 0.021476 | 0 | 1 |  |  | 0.0177618 | 0 |
| MAP | 0.873959 | 0.879314 | False | 0.544236 | 0.0206872 | 0 | 1 |  |  | 0.0177689 | 0 |
| DBP | 0.68596 | 0.688687 | False | 0.5 | 0.0181009 | 0 | 1 |  |  | 0.0177738 | 0 |
| Resp | 0.845653 | 0.850698 | False | 0.615109 | 0.0295925 | 0.196839 | 0.927763 | 0.0478296 | 0.0769591 | 0.0176997 | 0.0825565 |
| Lactate | 0.0266057 | 0.0267092 | True | 0.572751 | 0.0251803 | 0 | 1 |  |  | 0.0177453 | 0 |
| Creatinine | 0.0610853 | 0.0609994 | False | 0.551904 | 0.0220052 | 0 | 1 |  |  | 0.0177619 | 0 |
| WBC | 0.0641159 | 0.064142 | False | 0.545349 | 0.024182 | 0 | 1 |  |  | 0.0177507 | 0 |
| Platelets | 0.0595314 | 0.0593523 | False | 0.5 | 0.0181009 | 0 | 1 |  |  | 0.0177738 | 0 |
| Bilirubin_total | 0.0149604 | 0.0147676 | True | 0.598219 | 0.0285031 | 0 | 1 |  |  | 0.0177469 | 0 |
| Glucose | 0.171278 | 0.169371 | False | 0.5 | 0.0181009 | 0 | 1 |  |  | 0.0177738 | 0 |
| pH | 0.0693906 | 0.0696424 | False | 0.599836 | 0.0260765 | 0 | 1 |  |  | 0.01775 | 0 |
| BUN | 0.0687243 | 0.0688102 | False | 0.563741 | 0.0224797 | 0 | 1 |  |  | 0.0177506 | 0 |
| Hgb | 0.0739063 | 0.0734872 | False | 0.5 | 0.0181009 | 0 | 1 |  |  | 0.0177738 | 0 |
| Hct | 0.0885014 | 0.088853 | False | 0.5 | 0.0181009 | 0 | 1 |  |  | 0.0177738 | 0 |

## 3. Random subset results

Subsets are drawn from the 40 supported variables with seed 2019, before any split label is read. Six distinct subsets are scored at each size except 40, which is the complete set. Highest and lowest refer to utility inside that sample of subsets.

### validation

#### temporal_last

| k | subsets | mean AUROC | std AUROC | mean AUPRC | std AUPRC | mean utility | std utility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | 0.546768 | 0.0435022 | 0.0208345 | 0.00276268 | 0 | 0 |
| 2 | 6 | 0.539894 | 0.077977 | 0.0217626 | 0.00341992 | 0 | 0 |
| 3 | 6 | 0.575306 | 0.0318808 | 0.0231127 | 0.00343198 | 0 | 0 |
| 5 | 6 | 0.603909 | 0.0492859 | 0.0288762 | 0.00644133 | 0.0155083 | 0.035911 |
| 10 | 6 | 0.662753 | 0.0353144 | 0.0363476 | 0.00616631 | 0.048852 | 0.055658 |
| 20 | 6 | 0.734286 | 0.0414454 | 0.0562721 | 0.0173925 | 0.180528 | 0.0810348 |
| 30 | 6 | 0.80434 | 0.0475943 | 0.0861668 | 0.0230992 | 0.323869 | 0.116776 |
| 40 | 1 | 0.845345 | 0 | 0.11978 | 0 | 0.425821 | 0 |

k=1 highest utility in this sample: 0 with PaCO2.
k=1 lowest utility in this sample: 0 with Bilirubin_direct.
k=2 highest utility in this sample: 0 with HospAdmTime, PaCO2.
k=2 lowest utility in this sample: 0 with EtCO2, ICULOS.
k=3 highest utility in this sample: 0 with Bilirubin_direct, FiO2, Resp.
k=3 lowest utility in this sample: 0 with Glucose, Unit1, Unit2.
k=5 highest utility in this sample: 0.0887238 with ICULOS, Lactate, MAP, Temp, pH.
k=5 lowest utility in this sample: -4.89762e-05 with DBP, Hct, PaCO2, SaO2, Unit1.
k=10 highest utility in this sample: 0.139814 with Bilirubin_direct, Creatinine, FiO2, Glucose, MAP, Magnesium, Resp, Temp, Unit1, pH.
k=10 lowest utility in this sample: 0 with Bilirubin_direct, Chloride, Creatinine, HCO3, Hgb, ICULOS, Magnesium, PTT, TroponinI, Unit1.
k=20 highest utility in this sample: 0.329844 with AST, Age, Alkalinephos, Bilirubin_total, Calcium, Creatinine, DBP, FiO2, Fibrinogen, Gender, HCO3, Hct, ICULOS, Lactate, MAP, Phosphate, Platelets, Resp, SBP, WBC.
k=20 lowest utility in this sample: 0.110158 with Bilirubin_direct, Calcium, DBP, Fibrinogen, Glucose, HCO3, HR, Hgb, HospAdmTime, ICULOS, Lactate, O2Sat, PaCO2, Phosphate, Potassium, Resp, SBP, SaO2, Unit1, pH.
k=30 highest utility in this sample: 0.406759 with AST, Age, Alkalinephos, BaseExcess, Bilirubin_total, Calcium, Creatinine, FiO2, Fibrinogen, Gender, Glucose, HCO3, HR, Hct, HospAdmTime, ICULOS, Lactate, MAP, Magnesium, PTT, PaCO2, Phosphate, Resp, SBP, Temp, TroponinI, Unit1, Unit2, WBC, pH.
k=30 lowest utility in this sample: 0.0998134 with AST, Age, Alkalinephos, BUN, BaseExcess, Bilirubin_direct, Bilirubin_total, Calcium, Chloride, DBP, EtCO2, FiO2, Fibrinogen, Gender, HR, Hct, Hgb, HospAdmTime, Lactate, MAP, Magnesium, PTT, PaCO2, Phosphate, Platelets, Potassium, SBP, Unit1, Unit2, pH.
k=40 highest utility in this sample: 0.425821 with HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.
k=40 lowest utility in this sample: 0.425821 with HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.

#### flexible_xgboost

| k | subsets | mean AUROC | std AUROC | mean AUPRC | std AUPRC | mean utility | std utility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | 0.526116 | 0.0448369 | 0.0200434 | 0.00319549 | 0 | 0 |
| 2 | 6 | 0.583626 | 0.0751407 | 0.0303299 | 0.0180867 | 0.058342 | 0.115845 |
| 3 | 6 | 0.565157 | 0.0590979 | 0.023157 | 0.0059 | 0.0114631 | 0.0280789 |
| 5 | 6 | 0.678321 | 0.0687556 | 0.0512892 | 0.0328876 | 0.133335 | 0.150507 |
| 10 | 6 | 0.71129 | 0.0346085 | 0.0533596 | 0.0170351 | 0.205154 | 0.0890649 |
| 20 | 6 | 0.773038 | 0.0251276 | 0.0793955 | 0.0206747 | 0.303276 | 0.0587245 |
| 30 | 6 | 0.814827 | 0.025333 | 0.0986427 | 0.0199212 | 0.374583 | 0.0493456 |
| 40 | 1 | 0.837828 | 0 | 0.11268 | 0 | 0.415082 | 0 |

k=1 highest utility in this sample: 0 with PaCO2.
k=1 lowest utility in this sample: 0 with DBP.
k=2 highest utility in this sample: 0.289601 with EtCO2, ICULOS.
k=2 lowest utility in this sample: 0 with Glucose, Hct.
k=3 highest utility in this sample: 0.0687789 with Bilirubin_direct, FiO2, Resp.
k=3 lowest utility in this sample: 0 with Bilirubin_direct, Calcium, Fibrinogen.
k=5 highest utility in this sample: 0.354625 with ICULOS, Lactate, MAP, Temp, pH.
k=5 lowest utility in this sample: 0 with DBP, Hct, PaCO2, SaO2, Unit1.
k=10 highest utility in this sample: 0.317525 with Alkalinephos, Bilirubin_total, Chloride, Hct, HospAdmTime, ICULOS, Lactate, O2Sat, Phosphate, SBP.
k=10 lowest utility in this sample: 0.0867912 with Age, BaseExcess, DBP, HCO3, HR, Platelets, Resp, SBP, Unit1, pH.
k=20 highest utility in this sample: 0.355577 with AST, Age, Alkalinephos, Bilirubin_total, Calcium, Creatinine, DBP, FiO2, Fibrinogen, Gender, HCO3, Hct, ICULOS, Lactate, MAP, Phosphate, Platelets, Resp, SBP, WBC.
k=20 lowest utility in this sample: 0.218862 with AST, Alkalinephos, BaseExcess, Bilirubin_direct, Calcium, Creatinine, DBP, FiO2, Fibrinogen, Glucose, HCO3, HR, Hgb, HospAdmTime, Lactate, O2Sat, PaCO2, SBP, SaO2, Unit1.
k=30 highest utility in this sample: 0.410582 with AST, Age, Alkalinephos, BUN, BaseExcess, Chloride, Creatinine, DBP, EtCO2, FiO2, Glucose, HCO3, HR, HospAdmTime, ICULOS, MAP, O2Sat, PTT, PaCO2, Phosphate, Platelets, Potassium, Resp, SBP, SaO2, Temp, TroponinI, Unit1, WBC, pH.
k=30 lowest utility in this sample: 0.27651 with AST, Age, Alkalinephos, BUN, BaseExcess, Bilirubin_direct, Bilirubin_total, Calcium, Chloride, DBP, EtCO2, FiO2, Fibrinogen, Gender, HR, Hct, Hgb, HospAdmTime, Lactate, MAP, Magnesium, PTT, PaCO2, Phosphate, Platelets, Potassium, SBP, Unit1, Unit2, pH.
k=40 highest utility in this sample: 0.415082 with HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.
k=40 lowest utility in this sample: 0.415082 with HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.

### test

#### temporal_last

| k | subsets | mean AUROC | std AUROC | mean AUPRC | std AUPRC | mean utility | std utility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | 0.54 | 0.0349311 | 0.0202939 | 0.00220091 | 0 | 0 |
| 2 | 6 | 0.533961 | 0.0684172 | 0.0209922 | 0.00302761 | 0 | 0 |
| 3 | 6 | 0.571077 | 0.035977 | 0.0232492 | 0.00473626 | 0 | 0 |
| 5 | 6 | 0.599568 | 0.0436466 | 0.0282073 | 0.00520245 | 0.0124142 | 0.0278879 |
| 10 | 6 | 0.666349 | 0.028587 | 0.0356301 | 0.0037824 | 0.0468781 | 0.0503519 |
| 20 | 6 | 0.735174 | 0.0407204 | 0.059885 | 0.0204871 | 0.184021 | 0.0757958 |
| 30 | 6 | 0.798705 | 0.049324 | 0.0846997 | 0.024065 | 0.312848 | 0.122214 |
| 40 | 1 | 0.837347 | 0 | 0.115563 | 0 | 0.400773 | 0 |

k=1 highest utility in this sample: 0 with PaCO2.
k=1 lowest utility in this sample: 0 with Bilirubin_direct.
k=2 highest utility in this sample: 0 with HospAdmTime, PaCO2.
k=2 lowest utility in this sample: 0 with EtCO2, ICULOS.
k=3 highest utility in this sample: 0 with Bilirubin_direct, FiO2, Resp.
k=3 lowest utility in this sample: 0 with Glucose, Unit1, Unit2.
k=5 highest utility in this sample: 0.0691729 with ICULOS, Lactate, MAP, Temp, pH.
k=5 lowest utility in this sample: -1.41434e-05 with DBP, Hct, PaCO2, SaO2, Unit1.
k=10 highest utility in this sample: 0.118604 with Bilirubin_direct, Creatinine, FiO2, Glucose, MAP, Magnesium, Resp, Temp, Unit1, pH.
k=10 lowest utility in this sample: 0 with Bilirubin_direct, Chloride, Creatinine, HCO3, Hgb, ICULOS, Magnesium, PTT, TroponinI, Unit1.
k=20 highest utility in this sample: 0.32648 with AST, Age, Alkalinephos, Bilirubin_total, Calcium, Creatinine, DBP, FiO2, Fibrinogen, Gender, HCO3, Hct, ICULOS, Lactate, MAP, Phosphate, Platelets, Resp, SBP, WBC.
k=20 lowest utility in this sample: 0.128317 with AST, BaseExcess, Bilirubin_direct, Bilirubin_total, Chloride, FiO2, Gender, Hct, Hgb, HospAdmTime, MAP, PaCO2, Phosphate, Platelets, Resp, SBP, SaO2, TroponinI, Unit1, pH.
k=30 highest utility in this sample: 0.389991 with Age, Alkalinephos, BaseExcess, Bilirubin_direct, Bilirubin_total, Calcium, Chloride, DBP, FiO2, Fibrinogen, Gender, Glucose, HCO3, HR, Hct, Hgb, HospAdmTime, ICULOS, Lactate, MAP, Magnesium, PTT, PaCO2, Platelets, Potassium, Resp, SaO2, Temp, Unit1, WBC.
k=30 lowest utility in this sample: 0.0792249 with AST, Age, Alkalinephos, BUN, BaseExcess, Bilirubin_direct, Bilirubin_total, Calcium, Chloride, DBP, EtCO2, FiO2, Fibrinogen, Gender, HR, Hct, Hgb, HospAdmTime, Lactate, MAP, Magnesium, PTT, PaCO2, Phosphate, Platelets, Potassium, SBP, Unit1, Unit2, pH.
k=40 highest utility in this sample: 0.400773 with HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.
k=40 lowest utility in this sample: 0.400773 with HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.

#### flexible_xgboost

| k | subsets | mean AUROC | std AUROC | mean AUPRC | std AUPRC | mean utility | std utility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | 0.526934 | 0.043098 | 0.019913 | 0.00288593 | 0 | 0 |
| 2 | 6 | 0.571392 | 0.0708169 | 0.0294868 | 0.0190482 | 0.0539421 | 0.110953 |
| 3 | 6 | 0.56932 | 0.0635203 | 0.0238741 | 0.00753414 | 0.0212672 | 0.0520938 |
| 5 | 6 | 0.675193 | 0.065606 | 0.0482144 | 0.0290014 | 0.128183 | 0.137066 |
| 10 | 6 | 0.716968 | 0.0319355 | 0.0545042 | 0.0162452 | 0.201648 | 0.0774559 |
| 20 | 6 | 0.772959 | 0.024189 | 0.0844606 | 0.0263898 | 0.293142 | 0.0673207 |
| 30 | 6 | 0.813799 | 0.0205957 | 0.101907 | 0.0223562 | 0.357532 | 0.0449012 |
| 40 | 1 | 0.8341 | 0 | 0.121704 | 0 | 0.398744 | 0 |

k=1 highest utility in this sample: 0 with PaCO2.
k=1 lowest utility in this sample: 0 with DBP.
k=2 highest utility in this sample: 0.27723 with EtCO2, ICULOS.
k=2 lowest utility in this sample: 0 with Glucose, Hct.
k=3 highest utility in this sample: 0.127603 with Bilirubin_direct, FiO2, Resp.
k=3 lowest utility in this sample: 0 with Bilirubin_direct, Calcium, Fibrinogen.
k=5 highest utility in this sample: 0.326836 with ICULOS, Lactate, MAP, Temp, pH.
k=5 lowest utility in this sample: 0 with DBP, Hct, PaCO2, SaO2, Unit1.
k=10 highest utility in this sample: 0.290241 with Alkalinephos, Bilirubin_total, Chloride, Hct, HospAdmTime, ICULOS, Lactate, O2Sat, Phosphate, SBP.
k=10 lowest utility in this sample: 0.114507 with Age, BaseExcess, DBP, HCO3, HR, Platelets, Resp, SBP, Unit1, pH.
k=20 highest utility in this sample: 0.350046 with AST, Age, Alkalinephos, Bilirubin_total, Calcium, Creatinine, DBP, FiO2, Fibrinogen, Gender, HCO3, Hct, ICULOS, Lactate, MAP, Phosphate, Platelets, Resp, SBP, WBC.
k=20 lowest utility in this sample: 0.198546 with AST, Alkalinephos, BaseExcess, Bilirubin_direct, Calcium, Creatinine, DBP, FiO2, Fibrinogen, Glucose, HCO3, HR, Hgb, HospAdmTime, Lactate, O2Sat, PaCO2, SBP, SaO2, Unit1.
k=30 highest utility in this sample: 0.386429 with AST, Age, Alkalinephos, BUN, BaseExcess, Chloride, Creatinine, DBP, EtCO2, FiO2, Glucose, HCO3, HR, HospAdmTime, ICULOS, MAP, O2Sat, PTT, PaCO2, Phosphate, Platelets, Potassium, Resp, SBP, SaO2, Temp, TroponinI, Unit1, WBC, pH.
k=30 lowest utility in this sample: 0.268228 with AST, Age, Alkalinephos, BUN, BaseExcess, Bilirubin_direct, Bilirubin_total, Calcium, Chloride, DBP, EtCO2, FiO2, Fibrinogen, Gender, HR, Hct, Hgb, HospAdmTime, Lactate, MAP, Magnesium, PTT, PaCO2, Phosphate, Platelets, Potassium, SBP, Unit1, Unit2, pH.
k=40 highest utility in this sample: 0.398744 with HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.
k=40 lowest utility in this sample: 0.398744 with HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.

## 4. User-like scenarios

### validation

#### temporal_last

| scenario | variables | AUROC | AUPRC | sensitivity | specificity | precision | F1 | Brier | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_hr_only | 1 | 0.572457 | 0.0224077 | 0 | 1 |  |  | 0.0180268 | 0 |
| B_hr_resp | 2 | 0.604185 | 0.0261312 | 0 | 1 |  |  | 0.0180084 | 0 |
| C_hr_resp_temp | 3 | 0.645486 | 0.0321101 | 0.0168566 | 0.995192 | 0.0608919 | 0.0264039 | 0.0178861 | 0.00931713 |
| D_common_vitals | 6 | 0.654873 | 0.0364452 | 0.0197056 | 0.996303 | 0.0897297 | 0.0323146 | 0.0178869 | 0.0140437 |
| E_vitals_and_selected_labs | 12 | 0.686058 | 0.0423632 | 0.280627 | 0.895499 | 0.0473141 | 0.0809755 | 0.0176584 | 0.12531 |
| F_vitals_and_sparse_labs | 8 | 0.622106 | 0.0309839 | 0 | 1 |  |  | 0.0179982 | 0 |
| G_most_available | 30 | 0.833773 | 0.109606 | 0.555793 | 0.889532 | 0.0851273 | 0.147641 | 0.0168907 | 0.390993 |

#### flexible_xgboost

| scenario | variables | AUROC | AUPRC | sensitivity | specificity | precision | F1 | Brier | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_hr_only | 1 | 0.600681 | 0.0256545 | 0.173314 | 0.908988 | 0.0340199 | 0.0568757 | 0.0177889 | 0.0297557 |
| B_hr_resp | 2 | 0.624933 | 0.0284849 | 0.26472 | 0.87578 | 0.0379174 | 0.0663335 | 0.0177799 | 0.0719444 |
| C_hr_resp_temp | 3 | 0.664369 | 0.0391296 | 0.353514 | 0.869044 | 0.0475506 | 0.0838259 | 0.0176745 | 0.149321 |
| D_common_vitals | 6 | 0.6881 | 0.0452523 | 0.322887 | 0.894968 | 0.0537953 | 0.0922253 | 0.0176074 | 0.160091 |
| E_vitals_and_selected_labs | 12 | 0.675102 | 0.0375019 | 0.231244 | 0.925967 | 0.0546117 | 0.0883567 | 0.0176874 | 0.116017 |
| F_vitals_and_sparse_labs | 8 | 0.676761 | 0.0371239 | 0.287987 | 0.890744 | 0.0464822 | 0.0800449 | 0.0176944 | 0.118809 |
| G_most_available | 30 | 0.82506 | 0.108185 | 0.552707 | 0.889124 | 0.084409 | 0.146452 | 0.0168732 | 0.388505 |

### test

#### temporal_last

| scenario | variables | AUROC | AUPRC | sensitivity | specificity | precision | F1 | Brier | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_hr_only | 1 | 0.581058 | 0.0227242 | 0 | 1 |  |  | 0.0179682 | 0 |
| B_hr_resp | 2 | 0.622291 | 0.0301483 | 0 | 1 |  |  | 0.0179393 | 0 |
| C_hr_resp_temp | 3 | 0.649459 | 0.0316164 | 0.0208333 | 0.994027 | 0.0604167 | 0.0309829 | 0.0178308 | 0.0108425 |
| D_common_vitals | 6 | 0.667 | 0.0343908 | 0.0177203 | 0.996018 | 0.0758197 | 0.0287267 | 0.0178338 | 0.0109996 |
| E_vitals_and_selected_labs | 12 | 0.690675 | 0.0418689 | 0.296216 | 0.899025 | 0.0513044 | 0.0874607 | 0.0176072 | 0.14117 |
| F_vitals_and_sparse_labs | 8 | 0.642778 | 0.0324832 | 0 | 1 |  |  | 0.017936 | 0 |
| G_most_available | 30 | 0.824744 | 0.10664 | 0.523467 | 0.89352 | 0.0830958 | 0.143424 | 0.0168881 | 0.363193 |

#### flexible_xgboost

| scenario | variables | AUROC | AUPRC | sensitivity | specificity | precision | F1 | Brier | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_hr_only | 1 | 0.616393 | 0.0264752 | 0.182711 | 0.9049 | 0.034206 | 0.057624 | 0.0177284 | 0.0305937 |
| B_hr_resp | 2 | 0.647565 | 0.0338631 | 0.295977 | 0.87106 | 0.0405978 | 0.0714018 | 0.0176673 | 0.0936402 |
| C_hr_resp_temp | 3 | 0.679652 | 0.0419923 | 0.371887 | 0.86604 | 0.0486849 | 0.0860984 | 0.0175945 | 0.162625 |
| D_common_vitals | 6 | 0.701817 | 0.0453632 | 0.332854 | 0.891617 | 0.0535811 | 0.0923036 | 0.017548 | 0.162162 |
| E_vitals_and_selected_labs | 12 | 0.701122 | 0.046061 | 0.258381 | 0.923556 | 0.0586541 | 0.0956052 | 0.0175695 | 0.138607 |
| F_vitals_and_sparse_labs | 8 | 0.695493 | 0.0424615 | 0.318966 | 0.887309 | 0.0495905 | 0.0858358 | 0.0175967 | 0.142337 |
| G_most_available | 30 | 0.823651 | 0.111351 | 0.530412 | 0.893242 | 0.0839047 | 0.14489 | 0.0168328 | 0.370601 |

## 5. Generalization and dataset shift

The transfer experiments train on one hospital and test on the other. Thresholds were chosen on the source hospital's validation patients. These models are not the Phase 5 serving candidate. The drop is part of the result.

| experiment | external patients | hours | AUROC | AUPRC | utility | source validation utility |
| --- | --- | --- | --- | --- | --- | --- |
| Train A, test B | 20000 | 761995 | 0.785734 | 0.0613362 | 0.296388 | 0.421383 |
| Train B, test A | 20336 | 790215 | 0.757265 | 0.0800847 | 0.310152 | 0.395631 |

For scale, the pooled flexible model on the patient-level test split has AUROC 0.8341 and utility 0.398744. That test split mixes both hospitals, so it is not an external-hospital result.

Phase 4 contribution ranks put ICULOS first in both hospital test samples, with HospAdmTime and Unit1 also near the top of both. Large rank changes were concentrated in missingness indicators. A plausible reading is that some score comes from site measurement patterns as well as from physiology. That reading is not proof of the mechanism. It is a reason the pooled number cannot be described as external clinical generalization.

A later external validation set, from a system that did not contribute these files, is still required.

## 6. Calibration

Calibrators were fit on validation predictions. The test set was not used to choose Platt or isotonic.

| model | chosen | validation Brier Platt | validation Brier isotonic |
| --- | --- | --- | --- |
| temporal_last | isotonic | 0.0167803 | 0.016715 |
| flexible_xgboost | isotonic | 0.0168409 | 0.0167918 |

## 7. Data-sufficiency rule

Count vital and laboratory values that are finite at the scored hour. HIGH is 6 or more, MODERATE is 3 to 5, LOW is 1 or 2, and VERY_LOW is 0. Static fields, ICULOS, history length, and the risk probability do not change the band. This is a data-information label, not clinical confidence and not a diagnosis.

The response also returns `data_available`, `data_total`, `data_completeness`, `historical_hours`, `observed_dynamic_variables`, `observed_laboratory_variables`, and `has_prior_measurements`. Those counts are not folded into the risk probability.

## 8. Serving decision

Provisional research serving candidate: `flexible_xgboost`.

`flexible_xgboost` is the provisional research serving candidate because its validation application score is 0.292163. Full-input validation utility is 0.415082 for the selected model and 0.425821 for `temporal_last`. Mean validation utility on random subsets of size 5 and 10 is 0.169244 for the selected model and 0.0321802 for `temporal_last`. The application has to accept arbitrary parameter subsets, so that sparse term is half of the score.

Further external and clinical validation is still required. Phase 2 remains available through `artifacts/metadata/model_manifest.json`. The API reads `artifacts/metadata/phase5_model_manifest.json` and does not serve the Phase 2 model.

## 9. API architecture

FastAPI loads the manifest, booster, preprocessor, and calibrator once at process start. Routes call `src.inference.service.predict` with that loaded bundle. Feature construction stays in `src.inference`. The API does not rebuild features.

```
GET /health
GET /api/model-info
POST /api/predict
POST /api/predict-sequence
```

## 10. API request and response contract

The request contract is `docs/frontend_api_contract.md`. Omitted fields and null are missing. Numeric 0 is a measured zero. Unknown names, `SepsisLabel`, non-numeric values, duplicate hours, and non-chronological hours are rejected. The response probability is calibrated. `research_early_warning` is true when that probability is at least the validation threshold. That flag is not a diagnosis.

## 11. Test results

API tests call the saved Phase 5 model. Unit tests cover the selection rule, deterministic subsets, the information band, chronological rejection, and explanation grouping. `python -m pytest` passed 43 tests, including health, model info, sparse input, one-variable input, a fuller input, history, zero versus missing, unknown names, malformed bodies, and artifact checks.

## 12. Limitations

Hospital transfer utility is lower than pooled utility, and this phase does not remove that gap. Random subsets are a sample, not every combination. One-variable operating points use the full-input threshold, so sensitivity can be zero while AUROC is above 0.5. Very early alarms are not treated as a better clinical result. The information band is a count of present vital and laboratory values. It has not been checked against outcomes as a confidence measure. No frontend is included.
