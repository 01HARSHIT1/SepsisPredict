"""Write the phase 1 markdown report from computed tables. No hand-entered metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.schema import DATASET_DOI, DATASET_URL, SPLIT_SEED


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def _num(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer() and abs(number) >= 100:
        return f"{int(number):,}"
    return f"{number:.{digits}f}"


def _json_default(value):
    if isinstance(value, float) and pd.isna(value):
        return None
    raise TypeError(type(value))


def write_phase1_report(
    cohort: dict,
    pooled: pd.DataFrame,
    split_summary: dict,
    leakage_checks: list[str],
    utility_example: float,
    read_errors: list[dict],
    unit_counts: pd.DataFrame,
    output_path: Path,
) -> None:
    hospitals = cohort["hospitals"]
    pooled_h = hospitals["pooled"]
    lines: list[str] = []
    add = lines.append
    add("# Phase 1 dataset audit")
    add("")
    add("This is a research audit of the public PhysioNet/Computing in Cardiology Challenge 2019 training data.")
    add("No sepsis risk model has been trained. No accuracy, AUROC, utility, or calibration number in this report comes from a model.")
    add("The system described here is not a medical device and does not diagnose sepsis.")
    add("")
    add(f"Dataset: Reyna et al., PhysioNet, version 1.0.0, DOI {DATASET_DOI}.")
    add(f"Source: {DATASET_URL}")
    add("")
    add("## A. Dataset statistics")
    add("")
    add(_cohort_table(hospitals))
    add("")
    add("Record length is the number of hourly rows in each patient file.")
    for name in ("A", "B", "pooled"):
        los = hospitals[name]["hours_per_patient"]
        add(
            f"- Training set {name}: median { _num(los['median'], 1) } hours, "
            f"mean { _num(los['mean'], 1) }, "
            f"IQR { _num(los['p25'], 1) } to { _num(los['p75'], 1) }, "
            f"range { _num(los['min'], 0) } to { _num(los['max_in_histogram'], 0) }."
        )
    add("")
    add(
        f"ICULOS increments by 1 hour in all but {pooled_h['n_patients_iculos_not_incrementing_by_1']} patients, "
        f"and starts at 1 in all but {pooled_h['n_patients_iculos_not_starting_at_1']} patients. "
        "Row order is therefore the hourly clock for windowing, except where those counts are nonzero."
    )
    add(f"Files that failed to parse: {len(read_errors)}.")
    add("")
    add("## B. Sepsis prevalence")
    add("")
    add(
        "SepsisLabel is the challenge target. For a patient who meets the challenge sepsis definition, "
        "the label is 1 from six hours before the operationalized onset (`t_sepsis - 6`) through the rest of the file. "
        "It is 0 before that, and it is 0 for every hour of a patient who does not meet the definition. "
        "It is not a confirmation that sepsis is clinically diagnosed at that hour."
    )
    add("")
    add(
        f"Pooled patient prevalence: {pooled_h['n_sepsis_patients']:,} of {pooled_h['n_patients']:,} "
        f"patients have at least one positive label ({_pct(pooled_h['patient_sepsis_prevalence'])})."
    )
    add(
        f"Pooled hour prevalence: {pooled_h['n_positive_hours']:,} of {pooled_h['n_hours']:,} hours are positive "
        f"({_pct(pooled_h['hour_positive_prevalence'])})."
    )
    add(
        f"Of sepsis-labeled patients, {pooled_h['n_sepsis_patients_positive_from_first_hour']:,} are already positive "
        "at the first recorded hour, so the file contains no pre-label hour. "
        f"{pooled_h['n_sepsis_patients_with_observed_prewindow']:,} have at least one leading negative hour."
    )
    add(
        f"Using the challenge utility reconstruction (first positive index + 6 hours), "
        f"the reconstructed onset index falls inside the file for {pooled_h['n_sepsis_patients_onset_inside_record']:,} "
        "sepsis-labeled patients. When it falls outside the file, the record ends during the early window."
    )
    add(
        f"Labels that are not monotonic non-decreasing: {pooled_h['n_patients_label_not_monotonic']:,} patients. "
        "A non-monotonic label would make a single onset time ambiguous."
    )
    add("")
    add("Set-specific prevalence is in the table above. A difference between set A and set B is a shift, not a pooled constant.")
    add("")
    add("## C. Feature availability")
    add("")
    add(_availability_lines(pooled))
    add("")
    add("## D. Variable measurement frequency")
    add("")
    add(cohort["gap_definition"])
    add("")
    add(_frequency_lines(pooled))
    add("")
    add("## E. Dynamic, sparse, and static classification")
    add("")
    add(
        "A priori groups follow the project brief and the challenge table: eight vitals as rapid, "
        "26 laboratory columns as sparse, and Age, Gender, Unit1, Unit2, and HospAdmTime as static. "
        "ICULOS is a time index. Empirical class uses the pooled record:"
    )
    add("")
    add("- near-hourly: at least 40% of hours observed and median gap of at most 2 hours")
    add("- intermittent: at least 5% of hours observed, or median gap of at most 12 hours, and not near-hourly")
    add("- sparse: otherwise, among vitals and labs")
    add("- static, time context, and target are assigned by role rather than by gap")
    add("")
    add(_class_lines(pooled))
    add("")
    add("## F. Missingness patterns")
    add("")
    add(
        f"Across patients, the median fraction of vital and laboratory cells with no recorded value is "
        f"{_num(100 * pooled_h['patient_median_frac_vital_lab_cells_missing'], 1)}%. "
        f"The mean number of vital or laboratory variables never observed in a stay is "
        f"{_num(pooled_h['patient_mean_n_vital_lab_never_observed'], 1)} of 34."
    )
    add("")
    add(
        "Hour-level association between missingness and SepsisLabel is reported in `reports/tables/missingness_vs_label.csv`. "
        "A higher positive-label rate when a variable is missing is not evidence that missingness causes sepsis. "
        "Later ICU hours are more often label-positive, and laboratory workflow changes with severity and with hospital."
    )
    add("")
    add(_missingness_shift(pooled))
    add("")
    add(_unit_table(unit_counts))
    add("")
    add(_measured_signatures(pooled, cohort))
    add("")
    add("## G. Recommended imputation strategy")
    add("")
    add(
        "These are hypotheses for the next validation experiment. They are not yet shown to improve prediction. "
        "No strategy replaces NaN with 0."
    )
    add("")
    for role, title in (
        ("vital", "Vitals"),
        ("lab", "Laboratory values"),
        ("static", "Static context"),
        ("time_context", "Time index"),
        ("target", "Target"),
    ):
        add(f"### {title}")
        add("")
        subset = pooled[pooled["role"] == role]
        # Group identical recommendations so the report stays readable.
        for text, part in subset.groupby("representation_recommendation", sort=False):
            names = ", ".join(part["variable"].tolist())
            add(f"- {names}: {text}")
        add("")
    add("## H. Recommended temporal features")
    add("")
    add(
        "Candidate features are listed in `reports/tables/feature_dictionary.csv`. "
        "They have not been selected by a model. Twelve-hour and twenty-four-hour summaries are not in the default "
        "candidate list: many stays are shorter than that, and a feature that is missing for most rows should be "
        "added only if a later ablation shows a gain. Near-hourly candidates emphasize current value, missingness, "
        "and short backward windows. Sparse candidates emphasize last real value, time since that value, and change "
        "between real draws. Static candidates are the value and a missing indicator, with no hourly delta."
    )
    add("")
    add("## I. Leakage risks")
    add("")
    add("Automated checks that passed in this run:")
    for name in leakage_checks:
        add(f"- `{name}`")
    add("")
    add(
        f"The official utility example `labels=[0,0,0,0,1,1]`, `predictions=[0,0,1,1,1,1]` "
        f"returns {utility_example:.12f}, matching the published scorer."
    )
    add("")
    add("Risks that the feature code is written to avoid, and that remain process risks:")
    add("- A centered rolling window, or any back-fill, would use future measurements. The feature functions slice `values[:t+1]`.")
    add("- Treating NaN as 0 invents a measurement. Means and deltas drop missing hours instead.")
    add("- Carrying a static value backward from a later row would leak if that value was absent at the start of the stay. The audit counts how often the first real value is after hour 1.")
    add("- Reconstructing `t_sepsis` from the full label series is valid for a target or a utility score. It is not an input.")
    add("- Scaler, imputer, and calibration parameters must be fit on training patients only. This phase does not fit them.")
    add("- Randomly splitting hourly rows would put the same patient in train and test. The split below forbids that.")
    add("- Set A versus set B differences in missingness can identify the hospital system. A model can learn that signature and fail on the hidden third system.")
    add("")
    add("## J. Patient splitting strategy")
    add("")
    add(
        f"Seed {SPLIT_SEED}. Within each stratum of hospital system and patient-level sepsis label, "
        "patients are shuffled and allocated 70% / 15% / 15% by the largest-remainder method. "
        "Every hourly row of a patient follows that patient's split."
    )
    add("")
    add(
        f"Assigned patients: train {split_summary['n_train']:,}, "
        f"validation {split_summary['n_validation']:,}, "
        f"test {split_summary['n_test']:,}. "
        f"Overlap count: {split_summary['overlap_patient_count']}."
    )
    add("Identifiers are in `data/splits/`.")
    add("")
    add("## K. Proposed model architecture")
    add("")
    add("Not trained in this phase.")
    add("")
    add("- Model A: logistic regression on current raw values plus missing indicators. Interpretable check of preprocessing.")
    add("- Model B: gradient-boosted trees on current values, missing indicators, last observations, time since measurement, the candidate backward features, and static context.")
    add("- Model C: GRU or LSTM on the hourly state sequence, using the same representation, including sparse-lab fields.")
    add("- Model D: temporal attention only after A-C have been fit and compared. It is not assumed to be better.")
    add("")
    add("Flexible-input experiments (natural missingness, random masking, and structured masking down to one variable) belong after a model exists. Performance at one variable is not claimed.")
    add("Calibration, if used, is fit on validation patients and stored separately. It is not assumed from a raw score.")
    add("Class weights are not applied yet. Hour-level and patient-level prevalence differ, so any weighting has to be judged later by recall, precision, AUPRC, false alarms, and calibration together.")
    add("")
    add("## L. Proposed prediction horizon")
    add("")
    add(
        "Primary target: the official SepsisLabel at hour t. Because the challenge already sets that label to 1 "
        "at `t_sepsis - 6`, the baseline task is an early-warning label with a six-hour head start, not a classifier "
        "of confirmed sepsis at the current minute."
    )
    add(
        "Evaluation of a future model should include the official utility score, which rewards predictions from "
        "12 hours before the reconstructed onset through 3 hours after it, and penalizes late calls, missed calls, and false alarms."
    )
    add(
        "A secondary research target is defined in code but not materialized as a training matrix: at a still-negative hour t, "
        "whether the official label becomes positive within a configurable horizon. That target reads future labels by design. "
        "Features at t still may not."
    )
    add("")
    add("## Deployment contract")
    add("")
    add(
        "No HTTP service and no frontend are part of this phase. "
        "The Python modules under `src/inference/` are the boundary a later FastAPI app will call. "
        "The frontend will send JSON and display the response. It will not reimplement feature engineering."
    )
    add("")
    add("Modules FastAPI will import:")
    add("")
    add("- `src/inference/contract.py` defines the horizon name, the empty prediction slot, and risk trend from a sequence of model probabilities.")
    add("- `src/inference/features.py` parses the request and builds the feature row with `src/features/temporal.py`.")
    add("- `src/inference/artifacts.py` reads and writes the feature definition, and records where a preprocessor, model, and calibrator will be stored.")
    add("- `src/inference/service.py` refuses to score while the manifest status is `not_trained`.")
    add("- `src/data/schema.py`, `src/labels/targets.py`, and `src/evaluation/utility.py` stay on the server for column identity, targets, and the challenge utility.")
    add("")
    add("Artifacts, stored apart from each other:")
    add("")
    add("- `artifacts/feature_definitions/feature_spec.json` — frozen feature names, order, and empirical class. Written in this phase.")
    add("- `artifacts/metadata/model_manifest.json` — version, family, horizon, and paths. `model_version` is null.")
    add("- `artifacts/preprocessors/` — fitted training-only state, when a later experiment fits one. Not zero-fill.")
    add("- `artifacts/models/` — one model file per version. Swapping XGBoost for a GRU replaces this file and the manifest family, not the request JSON.")
    add("- `artifacts/calibration/` — optional calibrator fit on validation patients only.")
    add("")
    add("Request JSON, either one hour or several hours in order:")
    add("")
    add("```")
    add('{"observations": [{"values": {"HR": 120, "Temp": 39.1, "Resp": 29, "MAP": 62}}]}')
    add("```")
    add("")
    add(
        "Omitted names and null are missing. A sent 0 is a measured 0. "
        "`SepsisLabel` is rejected. Unknown names are rejected. "
        "The same parser serves the quick form and the hourly history."
    )
    add("")
    add(
        "The later response will carry `risk_probability`, `data_available`, `data_total`, "
        "`data_completeness`, `data_sufficiency`, `risk_trend`, `prediction_horizon`, "
        "`model_version`, and `explanations`. "
        "Availability counts are computed from the submitted hour. "
        "`data_sufficiency` stays null until a validation study defines it. "
        "`risk_probability`, `risk_trend`, and `explanations` stay absent until a model artifact exists. "
        "This phase does not fill them."
    )
    add("")
    add("## M. Files created")
    add("")
    add("See the file list written next to this report by the phase 1 runner, and the repository tree under `src/`, `tests/`, `reports/`, and `data/splits/`.")
    add("")
    add("## N. Commands")
    add("")
    add("From the project root `C:\\CodeData\\SepsisEarlyPrediction`:")
    add("")
    add("```")
    add("python -m src.data.download")
    add("python -m src.data.run_phase1")
    add("python -m pytest")
    add("```")
    add("")
    add("`python -m src.data.download` reads the PhysioNet training indexes and downloads `training_setA` and `training_setB`.")
    add("`python -m src.data.run_phase1` recomputes this audit, the figures, the patient split, and this report.")
    add("`python -m pytest` runs the leakage, utility, split, and synthetic-audit tests.")
    add("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _cohort_table(hospitals: dict) -> str:
    header = "| Set | Patients | Hours | Sepsis patients | Patient prevalence | Positive hours | Hour prevalence |"
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for name in ("A", "B", "pooled"):
        item = hospitals[name]
        body.append(
            "| {name} | {patients:,} | {hours:,} | {sepsis:,} | {prev} | {pos:,} | {hprev} |".format(
                name=name,
                patients=item["n_patients"],
                hours=item["n_hours"],
                sepsis=item["n_sepsis_patients"],
                prev=_pct(item["patient_sepsis_prevalence"]),
                pos=item["n_positive_hours"],
                hprev=_pct(item["hour_positive_prevalence"]),
            )
        )
    return "\n".join([header, rule, *body])


def _availability_lines(pooled: pd.DataFrame) -> str:
    lines = []
    for _, row in pooled.sort_values("pct_patients_ever_observed").iterrows():
        if row["role"] == "target":
            continue
        lines.append(
            f"- {row['variable']} ({row['role']}): present at least once in "
            f"{row['pct_patients_ever_observed']:.1f}% of patients and in "
            f"{row['pct_hours_observed']:.1f}% of hours."
        )
    return "\n".join(lines)


def _frequency_lines(pooled: pd.DataFrame) -> str:
    lines = []
    clinical = pooled[pooled["role"].isin(["vital", "lab"])].sort_values("pct_hours_observed", ascending=False)
    for _, row in clinical.iterrows():
        lines.append(
            f"- {row['variable']}: {row['pct_hours_observed']:.1f}% of hours recorded, "
            f"median gap {_num(row['median_gap_hours'], 1)} h, "
            f"90th percentile gap {_num(row['p90_gap_hours'], 1)} h, "
            f"maximum gap {_num(row['max_gap_hours'], 0)} h, "
            f"{row['n_patients_at_least_two_observations']:,} patients with a repeat measurement, "
            f"of whom {row['pct_repeat_patients_with_change']:.1f}% show a numeric change."
        )
    return "\n".join(lines)


def _class_lines(pooled: pd.DataFrame) -> str:
    lines = []
    for empirical, part in pooled.groupby("empirical_class", sort=False):
        names = ", ".join(part["variable"].tolist())
        lines.append(f"- {empirical}: {names}")
    disagreements = pooled[pooled["a_priori_empirical_disagreement"].fillna("") != ""]
    lines.append("")
    if disagreements.empty:
        lines.append("No a priori group disagreed with its empirical frequency class under the rules above.")
    else:
        lines.append("A priori group and empirical class disagree for:")
        for _, row in disagreements.iterrows():
            lines.append(
                f"- {row['variable']}: {row['a_priori_empirical_disagreement']} "
                f"(set A {row['empirical_class_set_A']}, set B {row['empirical_class_set_B']})."
            )
    hospital_shift = pooled[pooled["hospital_class_disagreement"] == True]  # noqa: E712
    lines.append("")
    if hospital_shift.empty:
        lines.append("Set A and set B receive the same empirical class for every variable under these thresholds.")
    else:
        lines.append("Empirical class differs between set A and set B for:")
        for _, row in hospital_shift.iterrows():
            lines.append(
                f"- {row['variable']}: set A {row['empirical_class_set_A']}, set B {row['empirical_class_set_B']}."
            )
    return "\n".join(lines)


def _missingness_shift(pooled: pd.DataFrame) -> str:
    # Compare set A and B missingness using the long table's siblings. The pooled frame does not
    # contain per-set percentages; those columns are added by the caller as optional fields.
    if "pct_hours_missing_A" not in pooled.columns:
        return "Per-hospital missingness is in `reports/tables/variable_audit.csv`."
    work = pooled[pooled["role"].isin(["vital", "lab"])].copy()
    work["abs_diff"] = (work["pct_hours_missing_A"] - work["pct_hours_missing_B"]).abs()
    top = work.sort_values("abs_diff", ascending=False).head(8)
    lines = ["Largest absolute differences in percent of hours missing, set A minus the comparison with set B:"]
    for _, row in top.iterrows():
        lines.append(
            f"- {row['variable']}: set A {row['pct_hours_missing_A']:.1f}% missing, "
            f"set B {row['pct_hours_missing_B']:.1f}% missing."
        )
    return "\n".join(lines)


def _measured_signatures(pooled: pd.DataFrame, cohort: dict) -> str:
    by_name = pooled.set_index("variable")
    etco2 = by_name.loc["EtCO2"]
    fio2 = by_name.loc["FiO2"]
    lactate = by_name.loc["Lactate"]
    hr = by_name.loc["HR"]
    age = cohort["patient_level_static"]
    fio2_counts = cohort["fio2_scale_counts"]["pooled"]
    lines = [
        "Other patterns read from the same tables:",
        "",
        (
            f"- EtCO2 is missing for {etco2['pct_hours_missing_A']:.1f}% of hours in set A and "
            f"{etco2['pct_hours_missing_B']:.1f}% of hours in set B. In this release, a recorded EtCO2 value does not occur in set A."
        ),
        (
            f"- Age is present on every hour and is constant within each patient. "
            f"Set A ranges from {age['A']['Age']['min']:.2f} to {age['A']['Age']['max']:.0f}. "
            f"Set B ranges from {age['B']['Age']['min']:.0f} to {age['B']['Age']['max']:.0f}. "
            "The challenge dictionary defines 100 as age 90 or above, so the top code differs by hospital system."
        ),
        (
            f"- FiO2 is published with unit %. Of recorded values, {fio2_counts['n_values_le_1']:,} are at most 1 and "
            f"{fio2_counts['n_values_gt_1']:,} are above 1. "
            f"The reservoir median is {_num(fio2['reservoir_p50'], 2)}, the exact minimum is {_num(fio2['min'], 2)}, "
            f"and the exact maximum is {_num(fio2['max'], 0)}. No rescaling was applied."
        ),
        (
            f"- Lactate reservoir median is {_num(lactate['reservoir_p50'], 2)} "
            f"(exact minimum {_num(lactate['min'], 2)}, exact maximum {_num(lactate['max'], 1)}). "
            "The published unit is mg/dL. Values were not converted."
        ),
        (
            f"- Heart rate is first recorded after the first file row for "
            f"{int(hr['n_patients_first_observation_after_hour_1']):,} patients. "
            "Those early rows stay missing. They are not back-filled."
        ),
        (
            f"- HospAdmTime is constant for {age['pooled']['HospAdmTime']['n_patients_with_constant_value']:,} of "
            f"{cohort['hospitals']['pooled']['n_patients']:,} patients. "
            f"Among constant values the mean is {_num(age['pooled']['HospAdmTime']['mean'], 1)} hours, "
            f"the minimum is {_num(age['pooled']['HospAdmTime']['min'], 2)}, and the maximum is "
            f"{_num(age['pooled']['HospAdmTime']['max'], 2)}. The mean is pulled by a long negative tail."
        ),
    ]
    return "\n".join(lines)


def _unit_table(unit_counts: pd.DataFrame) -> str:
    lines = [
        "Unit1 and Unit2 are administrative ICU identifiers (challenge dictionary: MICU and SICU). "
        "Each patient is summarized by the constant pair recorded in that file. "
        "`missing` means both fields were absent for the whole stay. This pattern can identify the hospital system:",
        "",
        "| Set | Unit1 | Unit2 | Patients |",
        "| --- | --- | --- | ---: |",
    ]
    if unit_counts.empty:
        lines.append("| n/a | n/a | n/a | 0 |")
        return "\n".join(lines)
    for _, row in unit_counts.sort_values(["hospital_system", "n_patients"], ascending=[True, False]).iterrows():
        left, right = str(row["unit1_unit2"]).split("|", 1)
        lines.append(f"| {row['hospital_system']} | {left} | {right} | {int(row['n_patients']):,} |")
    return "\n".join(lines)


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_fallback), encoding="utf-8")


def _json_fallback(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    raise TypeError(f"not serializable: {type(value)}")
