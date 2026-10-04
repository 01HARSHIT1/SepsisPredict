"""Phase 4 report. Numbers are copied from the checkpoint and the phase 2 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference.artifacts import project_root
from src.training.train import _sanitize


def write_phase4_report(root: Path, checkpoint: dict, spec: dict) -> None:
    phase2 = json.loads((root / "reports" / "phase2" / "checkpoint.json").read_text(encoding="utf-8"))
    _figures(root, checkpoint, phase2)
    text = _render(checkpoint, spec, phase2)
    path = root / "reports" / "phase4_report.md"
    path.write_text(text, encoding="utf-8")
    (root / "reports" / "phase4_metrics.json").write_text(
        json.dumps(_sanitize(checkpoint), indent=2), encoding="utf-8"
    )


def _render(checkpoint: dict, spec: dict, phase2: dict) -> str:
    lines = [
        "# Phase 4 research report",
        "",
        "This file is generated from `reports/phase4_metrics.json` and the unchanged Phase 2 checkpoint. "
        "`SepsisLabel` is the PhysioNet 2019 early-warning label. These experiments do not establish clinical efficacy. "
        "Count-based variable lists are availability-ranked feature masks: variables observed on the most training hours, "
        "not a ranking by predictive contribution. Age, Gender, and ICULOS lead that list because they are present on "
        "essentially every hour.",
        "",
        "## 1. Current baseline",
        "",
    ]
    lines.extend(_baseline(phase2))
    lines.extend(["", "## 2. Single flexible-input model", ""])
    lines.extend(_model_block(checkpoint.get("models", {}).get("flexible_xgboost"), checkpoint.get("masking_protocol")))
    lines.extend(["", "## 3. Flexible-input performance curve", ""])
    lines.extend(_curve(checkpoint))
    lines.extend(["", "## 4. Separate-model masking comparison", ""])
    lines.extend(_separate(phase2, checkpoint))
    lines.extend(["", "## 5. Hospital A to B", ""])
    lines.extend(_hospital(checkpoint, "hospital_A_to_B"))
    lines.extend(["", "## 6. Hospital B to A", ""])
    lines.extend(_hospital(checkpoint, "hospital_B_to_A"))
    lines.extend(["", "## 7. Administrative and site-feature ablations", ""])
    lines.extend(_ablation(checkpoint, phase2))
    lines.extend(["", "## 8. Feature importance stability", ""])
    lines.extend(_importance(checkpoint))
    lines.extend(["", "## 9. False alarm analysis", ""])
    lines.extend(_false_alarms(checkpoint))
    lines.extend(["", "## 10. Warning lead-time analysis", ""])
    lines.extend(_leads(checkpoint))
    lines.extend(["", "## 11. Threshold analysis", ""])
    lines.extend(_thresholds(checkpoint))
    lines.extend(["", "## 12. Calibration", ""])
    lines.extend(_calibration(checkpoint, phase2))
    lines.extend(["", "## 13. Temporal feature ablation", ""])
    lines.extend(_temporal(checkpoint, phase2))
    lines.extend(["", "## 14. GRU results", ""])
    lines.extend(_gru(checkpoint, phase2))
    lines.extend(["", "## 15. One-variable experiments", ""])
    lines.extend(_one_variable(checkpoint))
    lines.extend(["", "## 16. Provisional research serving candidate", ""])
    lines.extend(_recommendation(checkpoint))
    lines.extend(["", "## Inference", ""])
    lines.append("Phase 2 remains the default `python -m src.inference.score --request request.json` manifest.")
    lines.append("")
    lines.append("The provisional candidate is loaded with:")
    lines.append("")
    lines.append("```")
    lines.append("python -m src.inference.score --manifest artifacts/metadata/phase4_model_manifest.json --request request.json")
    lines.append("```")
    lines.append("")
    lines.append(
        "No FastAPI route and no frontend were added. GRU windows are left-padded when a stay is shorter than the window, "
        "so those hours are still scored and are not replaced with future measurements."
    )
    lines.append("")
    return "\n".join(lines)


def _baseline(phase2: dict) -> list[str]:
    model = phase2["models"]["xgboost_scale_pos_weight"]
    serving = phase2["serving"]
    return [
        "Phase 2 served model `xgboost_scale_pos_weight`, isotonic calibration, 311 features, 40 inputs. "
        "The test numbers below are the original Phase 2 checkpoint values.",
        "",
        _metrics_table(serving["test_calibrated"]),
        "",
        f"Phase 2 validation utility used for its own selection was {_num(serving['validation_calibrated_operating_point']['normalized_utility'])}. "
        f"Uncalibrated test AUROC was {_num(model['test_at_validation_threshold']['ranking']['auroc'])}.",
    ]


def _model_block(record: dict | None, protocol: dict | None) -> list[str]:
    if not record:
        return ["Not produced."]
    lines = [
        "One XGBoost model was trained on all 311 features. Each training row received one deterministic mask from seed 2019. "
        "Early stopping used unmasked validation rows. The saved threshold and calibrator were fit on those same unmasked validation predictions.",
        "",
    ]
    if protocol:
        lines.append(
            f"Mixture: natural {protocol['natural_fraction']}, "
            f"availability-weighted {protocol['availability_weighted_fraction']}, "
            f"uniform dropout {protocol['uniform_fraction']} at p={protocol['uniform_drop_probability']}, "
            "and the remainder structured. "
            + protocol["uniform_justification"]
            + " "
            + protocol["availability_weighted_rule"]
        )
        lines.append("")
    lines.append(_metrics_table(record["test_calibrated"]))
    lines.append("")
    lines.append(
        f"Uncalibrated test AUROC {_num(record['test_uncalibrated_ranking']['auroc'])}, "
        f"AUPRC {_num(record['test_uncalibrated_ranking']['auprc'])}, "
        f"Brier {_num(record['test_uncalibrated_ranking']['brier'])}."
    )
    lines.append("")
    lines.append(
        f"Validation calibrated threshold {_num(record['validation_calibrated_operating_point']['threshold'])}, "
        f"validation utility {_num(record['validation_calibrated_operating_point']['normalized_utility'])}."
    )
    lines.append("")
    lines.append(
        f"Runtime seconds: {_num(record.get('runtime_seconds'))}. "
        f"Process RSS after fitting: {_gb(record.get('process_peak_rss_bytes'))} GB."
    )
    return lines


def _curve(checkpoint: dict) -> list[str]:
    rows = checkpoint.get("flexible_curve", [])
    if not rows:
        return ["Not produced."]
    lines = [
        "The single flexible-input model is evaluated on the untouched pooled test set. "
        "Availability-ranked masks keep the same variables for every patient. "
        "The operating point uses the one threshold chosen on natural validation inputs. "
        "Performance is not expected to be equal across information levels.",
        "",
        "Figure: `reports/figures/phase4_information_curve.png`.",
        "",
        "| variables kept | hours | positive hours | negative hours | AUROC | AUPRC | Brier | sensitivity | specificity | precision | F1 | utility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        point = row["calibrated"]["operating_point"]
        ranking = row["calibrated"]["ranking"]
        lines.append(
            f"| {row['n_variables']} | {row['n_hours']} | {row['n_positive_hours']} | {row['n_negative_hours']} | "
            f"{_num(ranking['auroc'])} | {_num(ranking['auprc'])} | {_num(ranking['brier'])} | "
            f"{_num(point['sensitivity'])} | {_num(point['specificity'])} | {_num(point['precision'])} | "
            f"{_num(point['f1'])} | {_num(point['normalized_utility'])} |"
        )
    lines.append("")
    lines.append(
        "The one- and two-variable availability-ranked masks are Age, then Age and Gender. "
        "On this test set those inputs produced constant scores, so AUROC is 0.5 and the natural-validation threshold raised no alarms. "
        "The three-variable mask adds ICULOS, which is present every hour; that presence is not the same as clinical importance."
    )
    lines.append("")
    protocols = checkpoint.get("flexible_protocols", {})
    if protocols:
        lines.append("Additional test protocols on the same model, using evaluation seed 2024:")
        lines.append("")
        for name, row in protocols.items():
            ranking = row["calibrated"]["ranking"]
            point = row["calibrated"]["operating_point"]
            lines.append(
                f"- `{name}`: AUROC {_num(ranking['auroc'])}, AUPRC {_num(ranking['auprc'])}, "
                f"utility {_num(point['normalized_utility'])}, hours {row['n_hours']}, "
                f"positive {row['n_positive_hours']}, negative {row['n_negative_hours']}."
            )
    structured = checkpoint.get("flexible_structured", {})
    if structured:
        lines.append("")
        lines.append("Structured clinical-group masks on the same model:")
        lines.append("")
        for name, row in structured.items():
            ranking = row["calibrated"]["ranking"]
            point = row["calibrated"]["operating_point"]
            lines.append(
                f"- `{name}`: AUROC {_num(ranking['auroc'])}, AUPRC {_num(ranking['auprc'])}, "
                f"Brier {_num(ranking['brier'])}, sensitivity {_num(point['sensitivity'])}, "
                f"specificity {_num(point['specificity'])}, precision {_num(point['precision'])}, "
                f"F1 {_num(point['f1'])}, utility {_num(point['normalized_utility'])}."
            )
    return lines


def _separate(phase2: dict, checkpoint: dict) -> list[str]:
    lines = [
        "Separate-model masking baseline: each Phase 2 mask was its own retrained XGBoost. "
        "Those test metrics are cited from Phase 2 and were not recomputed. "
        "Names `top_N` in that checkpoint mean availability-ranked feature masks.",
        "",
        "| mask | variables | separate-model test AUROC | separate-model test utility | single-model test AUROC | single-model test utility |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    flexible = {row["n_variables"]: row for row in checkpoint.get("flexible_curve", [])}
    for name, record in phase2.get("masks", {}).items():
        if not name.startswith("top_"):
            continue
        count = record["n_variables"]
        single = flexible.get(count)
        single_auroc = _num(single["calibrated"]["ranking"]["auroc"]) if single else "null"
        single_utility = _num(single["calibrated"]["operating_point"]["normalized_utility"]) if single else "null"
        lines.append(
            f"| availability-ranked {count} | {count} | "
            f"{_num(record['test_at_validation_threshold']['ranking']['auroc'])} | "
            f"{_num(record['test_at_validation_threshold']['operating_point']['normalized_utility'])} | "
            f"{single_auroc} | {single_utility} |"
        )
    return lines


def _hospital(checkpoint: dict, name: str) -> list[str]:
    record = checkpoint.get("models", {}).get(name)
    if not record:
        return ["Not produced."]
    lines = [
        f"Train and validate on hospital {record['source_hospital']}. "
        f"Test on all patients from hospital {record['external_hospital']}. "
        f"{record['threshold_selected_on']}.",
        "",
        f"External patients {record['external_test_patients']}, hours {record['external_test_hours']}.",
        "",
        _metrics_table(record["test_calibrated"]),
        "",
        f"Source-hospital validation utility {_num(record['validation_calibrated_operating_point']['normalized_utility'])}.",
    ]
    return lines


def _ablation(checkpoint: dict, phase2: dict) -> list[str]:
    lines = [
        "Model A is the Phase 2 full model, cited above. The other rows are retrained on the pooled patient split. "
        "Removing a variable group measures association with the label in this dataset. It does not by itself decide that the group is unusable.",
        "",
        "| model | removed or retained | test AUROC | test AUPRC | test utility | validation utility |",
        "| --- | --- | --- | --- | --- | --- |",
        _ablation_row("A full Phase 2 model", "all 311 features", phase2["serving"]["test_calibrated"], phase2["serving"]["validation_calibrated_operating_point"]["normalized_utility"]),
    ]
    descriptions = {
        "ablation_no_admin": "drop Unit1, Unit2, HospAdmTime",
        "ablation_no_iculos": "drop ICULOS only",
        "ablation_physiology_age": "vitals, labs, and Age",
        "ablation_physiology": "vitals and labs only",
    }
    for name, text in descriptions.items():
        record = checkpoint.get("models", {}).get(name)
        if record:
            lines.append(_ablation_row(name, text, record["test_calibrated"], record["validation_calibrated_operating_point"]["normalized_utility"]))
    return lines


def _ablation_row(name, text, block, validation_utility) -> str:
    ranking = block["ranking"]
    point = block["operating_point"]
    return (
        f"| {name} | {text} | {_num(ranking['auroc'])} | {_num(ranking['auprc'])} | "
        f"{_num(point['normalized_utility'])} | {_num(validation_utility)} |"
    )


def _importance(checkpoint: dict) -> list[str]:
    block = checkpoint.get("importance")
    if not block:
        return ["Not produced."]
    lines = [
        "Mean absolute tree contributions on up to 3,000 test rows. Phase 2 model, split by hospital, and the physiology-plus-age ablation on pooled test rows.",
        "",
        "Features in the top 20 for both hospital A and hospital B test samples:",
        "",
    ]
    for feature in block["stability"]["stable_top_20_intersection"]:
        lines.append(f"- {feature}")
    lines.append("")
    lines.append("Largest absolute rank changes between hospital A and hospital B:")
    lines.append("")
    for row in block["stability"]["largest_rank_changes"][:10]:
        lines.append(
            f"- {row['feature']}: A rank {row['rank_hospital_A']}, B rank {row['rank_hospital_B']}, change {row['absolute_rank_change']}"
        )
    lines.append("")
    lines.append("Administrative or time features in the compared lists:")
    lines.append("")
    for row in block["stability"]["administrative_or_time_features"][:15]:
        lines.append(
            f"- {row['feature']}: A rank {row['rank_hospital_A']}, B rank {row['rank_hospital_B']}"
        )
    lines.append("")
    lines.append("Highest contributions, physiology-plus-age ablation, pooled test:")
    lines.append("")
    for row in block["physiology_age_ablation"][:8]:
        lines.append(f"- {row['feature']}: {_num(row['mean_absolute_contribution'])}")
    return lines


def _false_alarms(checkpoint: dict) -> list[str]:
    alarms = checkpoint.get("alarms", {})
    if not alarms:
        return ["Not produced."]
    lines = []
    for name, block in alarms.items():
        row = block["false_alarms"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"Threshold {_num(row['threshold'])}. Non-septic patients {row['n_nonseptic_patients']}, hours {row['n_nonseptic_hours']}.",
                f"False-positive hours {row['false_positive_hours']} "
                f"({_num(row['false_positive_hours_per_nonseptic_patient'])} per non-septic patient, "
                f"{_num(row['false_positive_hours_per_nonseptic_patient_day'])} per non-septic patient-day).",
                f"Fraction of non-septic patients ever alarmed: {_num(row['fraction_nonseptic_patients_ever_alarmed'])}.",
                f"False-alarm episodes {row['n_false_alarm_episodes']}, mean duration {_num(row['mean_false_alarm_episode_hours'])} hours.",
                f"Persistent episodes (at least 3 hours): {row['n_persistent_false_alarm_episodes']} "
                f"({_num(row['fraction_false_alarm_episodes_persistent'])} of episodes).",
                "",
            ]
        )
    return lines


def _leads(checkpoint: dict) -> list[str]:
    alarms = checkpoint.get("alarms", {})
    if not alarms:
        return ["Not produced."]
    lines = [
        "Lead time is reconstructed onset minus the first alarm. A long lead is reported as a bin, not as a better result. "
        "Figure: `reports/figures/phase4_lead_time.png`.",
        "",
    ]
    for name, block in alarms.items():
        lead = block["lead_time"]
        stats = lead["lead_hours"]
        fractions = lead["bin_fractions_among_onset_inside"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"Onset inside the file: {lead['n_onset_inside_file']}. Alarmed: {lead['n_with_alarm']}.",
                f"Median {_num(stats['median'])}, mean {_num(stats['mean'])}, P25 {_num(stats['p25'])}, "
                f"P75 {_num(stats['p75'])}, minimum {_num(stats['minimum'])}, maximum {_num(stats['maximum'])}.",
                f"More than 24 hours before onset: {_num(fractions['more_than_24h_before_onset'])}.",
                f"12 to 24 hours before onset: {_num(fractions['from_12h_to_24h_before_onset'])}.",
                f"6 to 12 hours before onset: {_num(fractions['from_6h_to_12h_before_onset'])}.",
                f"Inside the official six-hour window: {_num(fractions['inside_official_six_hour_window'])}.",
                f"After reconstructed onset: {_num(fractions['after_reconstructed_onset'])}.",
                f"Never alarmed: {_num(fractions['never_alarmed'])}.",
                f"Among alarmed patients, fraction whose first alarm is before the official positive label: "
                f"{_num(lead['fraction_of_alarmed_before_official_positive_window'])}.",
                f"Among alarmed patients, fraction still alarming at the last observed hour: "
                f"{_num(lead['fraction_of_alarmed_still_positive_at_record_end'])}.",
                "",
            ]
        )
    return lines


def _thresholds(checkpoint: dict) -> list[str]:
    tables = checkpoint.get("threshold_analysis", {})
    if not tables:
        return ["Not produced."]
    lines = [tables.get("selection_rule", ""), "", "Validation rows only. Brier does not depend on the threshold; it is in the calibration section.", ""]
    for name in ("phase2_served_model", "flexible_xgboost"):
        rows = tables.get(name, [])
        if not rows:
            continue
        chosen = max(rows, key=lambda row: (row["normalized_utility"], -row["threshold"]))
        lines.append(
            f"`{name}` validation utility is highest at threshold {_num(chosen['threshold'])} "
            f"(utility {_num(chosen['normalized_utility'])}, sensitivity {_num(chosen['sensitivity'])}, "
            f"precision {_num(chosen['precision'])}, "
            f"non-septic patients ever alarmed {_num(chosen['fraction_nonseptic_patients_ever_alarmed'])})."
        )
        lines.append("")
    lines.append("The full 50-point validation tables are in `reports/phase4_metrics.json`.")
    return lines


def _calibration(checkpoint: dict, phase2: dict) -> list[str]:
    record = checkpoint.get("models", {}).get("flexible_xgboost")
    if not record:
        return ["Not produced."]
    phase2_calibration = phase2["serving"]["calibration"]
    lines = [
        "Calibrators are fit on validation predictions. The plotted curves use the untouched test set. "
        "Figure: `reports/figures/phase4_calibration_curve.png`.",
        "",
        f"Phase 2 chosen calibrator `{phase2_calibration['chosen']}`. "
        f"Validation Brier Platt {_num(phase2_calibration['validation_brier_platt'])}, "
        f"isotonic {_num(phase2_calibration['validation_brier_isotonic'])}. "
        f"Test Brier {_num(phase2['serving']['test_calibrated']['ranking']['brier'])}.",
        "",
        f"Flexible model chosen calibrator `{record['calibration']['chosen']}`. "
        f"Validation Brier Platt {_num(record['calibration']['validation_brier_platt'])}, "
        f"isotonic {_num(record['calibration']['validation_brier_isotonic'])}. "
        f"Uncalibrated test Brier {_num(record['test_uncalibrated_ranking']['brier'])}. "
        f"Calibrated test Brier {_num(record['test_calibrated']['ranking']['brier'])}.",
        "",
        "Flexible-model calibrated test reliability:",
        "",
        "| bin | n | mean predicted | fraction positive |",
        "| --- | --- | --- | --- |",
    ]
    for row in record["test_calibrated"]["reliability"]:
        lines.append(
            f"| {row['bin']} | {row['n']} | {_num(row['mean_predicted'])} | {_num(row['fraction_positive'])} |"
        )
    return lines


def _temporal(checkpoint: dict, phase2: dict) -> list[str]:
    lines = [
        "Tree models retrained on nested feature groups. D is the Phase 2 full feature set, cited rather than retrained.",
        "",
        "| representation | features | test AUROC | test AUPRC | test utility | validation utility |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    order = (
        ("temporal_current", "A current values"),
        ("temporal_missing", "B current plus missingness"),
        ("temporal_last", "C plus last sparse value and time since measurement"),
    )
    for key, label in order:
        record = checkpoint.get("models", {}).get(key)
        if not record:
            continue
        ranking = record["test_calibrated"]["ranking"]
        point = record["test_calibrated"]["operating_point"]
        lines.append(
            f"| {label} | {len(record['feature_names'])} | {_num(ranking['auroc'])} | {_num(ranking['auprc'])} | "
            f"{_num(point['normalized_utility'])} | {_num(record['validation_calibrated_operating_point']['normalized_utility'])} |"
        )
    full = phase2["serving"]["test_calibrated"]
    lines.append(
        f"| D full temporal feature set, Phase 2 | 311 | {_num(full['ranking']['auroc'])} | {_num(full['ranking']['auprc'])} | "
        f"{_num(full['operating_point']['normalized_utility'])} | {_num(phase2['serving']['validation_calibrated_operating_point']['normalized_utility'])} |"
    )
    return lines


def _gru(checkpoint: dict, phase2: dict) -> list[str]:
    gru = checkpoint.get("gru", {})
    if not gru:
        return ["Not produced."]
    logistic = phase2["models"]["logistic_unweighted"]["test_at_validation_threshold"]
    ranking = logistic["ranking"]
    point = logistic["operating_point"]
    lines = [
        "Logistic metrics are cited from the Phase 2 checkpoint and were not refit. "
        "The logistic threshold was chosen on uncalibrated validation utility.",
        "",
        f"Logistic unweighted test: AUROC {_num(ranking['auroc'])}, AUPRC {_num(ranking['auprc'])}, "
        f"Brier {_num(ranking['brier'])}, sensitivity {_num(point['sensitivity'])}, "
        f"specificity {_num(point['specificity'])}, precision {_num(point['precision'])}, "
        f"F1 {_num(point['f1'])}, utility {_num(point['normalized_utility'])}, "
        f"calibration slope {_num(logistic['calibration_line']['slope'])}.",
        "",
        "Each GRU window is left-padded for short stays, so hours are not dropped. "
        "Coverage counts how many training hours lack a full window. "
        "Measured values are scaled with the training mean and standard deviation and then filled with 0 only beside a missingness indicator. "
        "That fill is not a fresh measurement.",
        "",
        "| window | hours padded | fraction padded | test AUROC | test AUPRC | test Brier | test utility | validation utility | runtime seconds | process RSS GB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, record in gru.items():
        coverage = record["coverage"]
        ranking = record["test_calibrated"]["ranking"]
        point = record["test_calibrated"]["operating_point"]
        lines.append(
            f"| {key} | {coverage['hours_with_left_padding']} | {_num(coverage['fraction_hours_with_left_padding'])} | "
            f"{_num(ranking['auroc'])} | {_num(ranking['auprc'])} | {_num(ranking['brier'])} | "
            f"{_num(point['normalized_utility'])} | {_num(record['validation_calibrated_operating_point']['normalized_utility'])} | "
            f"{_num(record['runtime_seconds'])} | {_gb(record.get('process_rss_bytes'))} |"
        )
    if checkpoint.get("gru_validation_selection"):
        selected = checkpoint["gru_validation_selection"]
        lines.extend(["", f"GRU window retained for the candidate comparison: `{selected['model_name']}`."])
    return lines


def _one_variable(checkpoint: dict) -> list[str]:
    rows = checkpoint.get("flexible_one_variable", [])
    if not rows:
        return ["Not produced."]
    lines = [
        "Single flexible-input model, one retained variable, untouched test set, natural missingness inside that variable. "
        "These are not clinical-reliability claims. `data_sufficiency` stays null. "
        "Sensitivity of 0 means no test hour reached the threshold chosen on full natural validation inputs. "
        "AUROC can still sit above 0.5 when the scores order hours but remain below that threshold.",
        "",
        "| variable | hours | positive | negative | AUROC | AUPRC | Brier | sensitivity | specificity | precision | F1 | utility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        point = row["calibrated"]["operating_point"]
        ranking = row["calibrated"]["ranking"]
        lines.append(
            f"| {row['variable']} | {row['n_hours']} | {row['n_positive_hours']} | {row['n_negative_hours']} | "
            f"{_num(ranking['auroc'])} | {_num(ranking['auprc'])} | {_num(ranking['brier'])} | "
            f"{_num(point['sensitivity'])} | {_num(point['specificity'])} | {_num(point['precision'])} | "
            f"{_num(point['f1'])} | {_num(point['normalized_utility'])} |"
        )
    return lines


def _recommendation(checkpoint: dict) -> list[str]:
    recommendation = checkpoint.get("recommendation")
    if not recommendation:
        return ["Not produced."]
    lines = [
        f"Provisional research serving candidate: `{recommendation['model_name']}` ({recommendation['model_family']}).",
        "",
        recommendation["selection_rule"],
        "",
        "Further external and clinical validation is still required.",
        "",
        "The single flexible-input model is `flexible_xgboost`. "
        "It is not this provisional candidate: its pooled validation utility is lower than `temporal_last` and the Phase 2 model. "
        "It remains saved under `artifacts/models/phase4_flexible_xgboost/`.",
        "",
        "| candidate | family | flexible input | validation utility | validation AUPRC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in recommendation["candidates"]:
        lines.append(
            f"| {row['model_name']} | {row['family']} | {row['flexible_input']} | "
            f"{_num(row['validation_utility'])} | {_num(row['validation_auprc'])} |"
        )
    return lines


def _gb(value) -> str:
    if value is None:
        return ""
    return _num(float(value) / 1_000_000_000)


def _metrics_table(block: dict) -> str:
    ranking = block["ranking"]
    point = block["operating_point"]
    line = block["calibration_line"]
    negative = ranking["n_hours"] - ranking["n_positive_hours"]
    return "\n".join(
        [
            "| metric | value |",
            "| --- | --- |",
            f"| hours | {ranking['n_hours']} |",
            f"| positive hours | {ranking['n_positive_hours']} |",
            f"| negative hours | {negative} |",
            f"| AUROC | {_num(ranking['auroc'])} |",
            f"| AUPRC | {_num(ranking['auprc'])} |",
            f"| Brier | {_num(ranking['brier'])} |",
            f"| calibration slope | {_num(line['slope'])} |",
            f"| calibration intercept | {_num(line['intercept'])} |",
            f"| threshold | {_num(point['threshold'])} |",
            f"| sensitivity | {_num(point['sensitivity'])} |",
            f"| specificity | {_num(point['specificity'])} |",
            f"| precision | {_num(point['precision'])} |",
            f"| F1 | {_num(point['f1'])} |",
            f"| normalized utility | {_num(point['normalized_utility'])} |",
        ]
    )


def _figures(root: Path, checkpoint: dict, phase2: dict) -> None:
    figure_dir = root / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    curve = checkpoint.get("flexible_curve", [])
    if curve:
        separate = []
        for name, record in phase2.get("masks", {}).items():
            if name.startswith("top_"):
                separate.append(
                    (
                        record["n_variables"],
                        record["test_at_validation_threshold"]["ranking"]["auroc"],
                        record["test_at_validation_threshold"]["ranking"]["auprc"],
                    )
                )
        separate.sort()
        figure, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot([row["n_variables"] for row in curve], [row["calibrated"]["ranking"]["auroc"] for row in curve], marker="o", label="single flexible-input model")
        if separate:
            axes[0].plot([item[0] for item in separate], [item[1] for item in separate], marker="s", label="separate-model masking baseline")
        axes[0].set_xlabel("availability-ranked variables kept")
        axes[0].set_ylabel("test AUROC")
        axes[0].legend()
        axes[1].plot([row["n_variables"] for row in curve], [row["calibrated"]["ranking"]["auprc"] for row in curve], marker="o", label="single flexible-input model")
        if separate:
            axes[1].plot([item[0] for item in separate], [item[2] for item in separate], marker="s", label="separate-model masking baseline")
        axes[1].set_xlabel("availability-ranked variables kept")
        axes[1].set_ylabel("test AUPRC")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(figure_dir / "phase4_information_curve.png", dpi=120)
        plt.close(figure)
    record = checkpoint.get("models", {}).get("flexible_xgboost")
    if record:
        figure, axis = plt.subplots(figsize=(5, 5))
        reliable = record["test_calibrated"]["reliability"]
        axis.plot([row["mean_predicted"] for row in reliable], [row["fraction_positive"] for row in reliable], marker="o", label="flexible model")
        phase2_bins = phase2["serving"]["test_calibrated"]["reliability"]
        axis.plot([row["mean_predicted"] for row in phase2_bins], [row["fraction_positive"] for row in phase2_bins], marker="s", label="phase 2 model")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set_xlabel("mean predicted probability")
        axis.set_ylabel("fraction of positive hours")
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / "phase4_calibration_curve.png", dpi=120)
        plt.close(figure)
    alarms = checkpoint.get("alarms", {})
    if alarms:
        figure, axis = plt.subplots(figsize=(7, 4))
        for name, block in alarms.items():
            histogram = block["lead_time"].get("lead_histogram")
            if not histogram:
                continue
            edges = histogram["bin_edges_hours"]
            counts = histogram["counts"]
            centers = [(edges[index] + edges[index + 1]) / 2 for index in range(len(counts))]
            axis.plot(centers, counts, marker="o", label=name)
        axis.set_xlabel("first-alarm lead hours")
        axis.set_ylabel("sepsis patients with onset inside the file")
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / "phase4_lead_time.png", dpi=120)
        plt.close(figure)


def _num(value) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6g}"
