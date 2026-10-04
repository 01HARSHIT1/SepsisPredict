"""Phase 5 report. Numbers come from the phase 5 checkpoint and the phase 4 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from src.inference.sufficiency import RULE_TEXT
from src.training.train import _sanitize


def write_phase5_report(root: Path, checkpoint: dict) -> None:
    phase4 = json.loads((root / "reports" / "phase4" / "checkpoint.json").read_text(encoding="utf-8"))
    text = _render(checkpoint, phase4)
    (root / "reports" / "phase5_report.md").write_text(text, encoding="utf-8")
    (root / "reports" / "phase5_metrics.json").write_text(json.dumps(_sanitize(checkpoint), indent=2), encoding="utf-8")


def _render(checkpoint: dict, phase4: dict) -> str:
    decision = checkpoint["decision"]
    lines = [
        "# Phase 5 research report",
        "",
        "This is a research early-warning model based on the PhysioNet 2019 SepsisLabel. "
        "It is not a clinical diagnosis system. No new model family was trained. "
        "The serving choice uses validation rows only. Test rows are reported after that choice.",
        "",
        "## 1. Final candidate comparison",
        "",
        decision["rule"],
        "",
        "| model | validation utility | validation AUPRC | validation AUROC | sparse utility k=5 and k=10 | application score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in decision["candidates"]:
        lines.append(
            f"| {row['model_name']} | {_num(row['full_utility'])} | {_num(row['full_auprc'])} | "
            f"{_num(row['full_auroc'])} | {_num(row['sparse_utility'])} | {_num(row['application_score'])} |"
        )
    lines.extend(["", "### Validation hospital slice of the pooled models", ""])
    lines.append("These rows score the pooled model on validation patients from one hospital. They are not a retrained transfer experiment.")
    lines.append("")
    lines.append("| model | hospital | hours | AUROC | AUPRC | utility |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in ("temporal_last", "flexible_xgboost"):
        for hospital, metrics in checkpoint["validation"][name]["hospital_slice"].items():
            lines.append(
                f"| {name} | {hospital} | {metrics['n_hours']} | {_num(metrics['auroc'])} | "
                f"{_num(metrics['auprc'])} | {_num(metrics['utility'])} |"
            )
    lines.extend(["", "### Test evaluation after the validation choice", ""])
    lines.append("| model | test AUROC | test AUPRC | test utility | flexible input | feature count |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in ("temporal_last", "flexible_xgboost"):
        metrics = checkpoint["test"][name]["full"]
        record = phase4["models"][name]
        lines.append(
            f"| {name} | {_num(metrics['auroc'])} | {_num(metrics['auprc'])} | {_num(metrics['utility'])} | "
            f"{name == 'flexible_xgboost'} | {len(record['feature_names'])} |"
        )
    lines.extend(
        [
            "",
            "Both candidates are histogram XGBoost models with isotonic calibration and the same canonical feature builder. "
            f"`temporal_last` scores {len(phase4['models']['temporal_last']['feature_names'])} columns. "
            f"`flexible_xgboost` scores {len(phase4['models']['flexible_xgboost']['feature_names'])} columns. "
            "Explanations for both are summed pre-calibration tree contributions grouped by clinical variable. "
            "The flexible model was trained with a deterministic mixture of natural and artificial missingness. "
            "`temporal_last` was not.",
            "",
            "## 2. One-variable results",
            "",
            "Each row keeps one variable and its natural missingness inside the single flexible-input model. "
            "The threshold is that model's natural-input validation threshold. "
            "A positive result here is not a claim that the variable can identify sepsis.",
            "",
            "### Validation",
            "",
            *_one_variable_table(checkpoint["validation"]["flexible_xgboost"]["one_variable"]),
            "",
            "### Test",
            "",
            *_one_variable_table(checkpoint["test"]["flexible_xgboost"]["one_variable"]),
            "",
            "## 3. Random subset results",
            "",
            "Subsets are drawn from the 40 supported variables with seed 2019, before any split label is read. "
            "Six distinct subsets are scored at each size except 40, which is the complete set. "
            "Highest and lowest refer to utility inside that sample of subsets.",
            "",
        ]
    )
    for split_name in ("validation", "test"):
        lines.extend([f"### {split_name}", ""])
        for name in ("temporal_last", "flexible_xgboost"):
            lines.extend([f"#### {name}", ""])
            lines.extend(_subset_table(checkpoint[split_name][name]["random_subsets"]))
            lines.append("")
    lines.extend(["## 4. User-like scenarios", ""])
    for split_name in ("validation", "test"):
        lines.extend([f"### {split_name}", ""])
        for name in ("temporal_last", "flexible_xgboost"):
            lines.extend([f"#### {name}", ""])
            lines.extend(_scenario_table(checkpoint[split_name][name]["scenarios"]))
            lines.append("")
    lines.extend(["## 5. Generalization and dataset shift", ""])
    lines.extend(_hospital_transfer(phase4))
    lines.extend(["", "## 6. Calibration", ""])
    lines.extend(_calibration(phase4))
    lines.extend(
        [
            "",
            "## 7. Data-sufficiency rule",
            "",
            RULE_TEXT,
            "",
            "The response also returns `data_available`, `data_total`, `data_completeness`, "
            "`historical_hours`, `observed_dynamic_variables`, `observed_laboratory_variables`, "
            "and `has_prior_measurements`. Those counts are not folded into the risk probability.",
            "",
            "## 8. Serving decision",
            "",
        f"Provisional research serving candidate: `{decision['model_name']}`.",
        "",
        _decision_reason(decision),
        "",
        "Further external and clinical validation is still required. "
            "Phase 2 remains available through `artifacts/metadata/model_manifest.json`. "
            "The API reads `artifacts/metadata/phase5_model_manifest.json` and does not serve the Phase 2 model.",
            "",
            "## 9. API architecture",
            "",
            "FastAPI loads the manifest, booster, preprocessor, and calibrator once at process start. "
            "Routes call `src.inference.service.predict` with that loaded bundle. "
            "Feature construction stays in `src.inference`. The API does not rebuild features.",
            "",
            "```",
            "GET /health",
            "GET /api/model-info",
            "POST /api/predict",
            "POST /api/predict-sequence",
            "```",
            "",
            "## 10. API request and response contract",
            "",
            "The request contract is `docs/frontend_api_contract.md`. "
            "Omitted fields and null are missing. Numeric 0 is a measured zero. "
            "Unknown names, `SepsisLabel`, non-numeric values, duplicate hours, and non-chronological hours are rejected. "
            "The response probability is calibrated. `research_early_warning` is true when that probability "
            "is at least the validation threshold. That flag is not a diagnosis.",
            "",
            "## 11. Test results",
            "",
            "API tests call the saved Phase 5 model. Unit tests cover the selection rule, deterministic subsets, "
            "the information band, chronological rejection, and explanation grouping. "
            "`python -m pytest` passed 43 tests, including health, model info, sparse input, one-variable input, "
            "a fuller input, history, zero versus missing, unknown names, malformed bodies, and artifact checks.",
            "",
            "## 12. Limitations",
            "",
            "Hospital transfer utility is lower than pooled utility, and this phase does not remove that gap. "
            "Random subsets are a sample, not every combination. "
            "One-variable operating points use the full-input threshold, so sensitivity can be zero while AUROC is above 0.5. "
            "Very early alarms are not treated as a better clinical result. "
            "The information band is a count of present vital and laboratory values. "
            "It has not been checked against outcomes as a confidence measure. "
            "No frontend is included.",
            "",
        ]
    )
    return "\n".join(lines)


def _one_variable_table(rows: list[dict]) -> list[str]:
    lines = [
        "| variable | training observed | split observed | sparse | AUROC | AUPRC | sensitivity | specificity | precision | F1 | Brier | utility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variable']} | {_num(row['training_observation_rate'])} | {_num(row['split_observation_rate'])} | "
            f"{row['too_sparse_for_stable_operating_point']} | {_num(row['auroc'])} | {_num(row['auprc'])} | "
            f"{_num(row['sensitivity'])} | {_num(row['specificity'])} | {_num(row['precision'])} | {_num(row['f1'])} | "
            f"{_num(row['brier'])} | {_num(row['utility'])} |"
        )
    return lines


def _subset_table(blocks: dict) -> list[str]:
    lines = [
        "| k | subsets | mean AUROC | std AUROC | mean AUPRC | std AUPRC | mean utility | std utility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    notes = []
    for size in sorted(blocks, key=lambda value: int(value)):
        block = blocks[size]
        lines.append(
            f"| {size} | {block['n_subsets']} | {_num(block['mean_auroc'])} | {_num(block['std_auroc'])} | "
            f"{_num(block['mean_auprc'])} | {_num(block['std_auprc'])} | {_num(block['mean_utility'])} | {_num(block['std_utility'])} |"
        )
        high = block["highest_utility_subset"]
        low = block["lowest_utility_subset"]
        notes.append(
            f"k={size} highest utility in this sample: {_num(high['utility'])} with {', '.join(high['variables'])}."
        )
        notes.append(
            f"k={size} lowest utility in this sample: {_num(low['utility'])} with {', '.join(low['variables'])}."
        )
    return lines + [""] + notes


def _scenario_table(rows: dict) -> list[str]:
    lines = [
        "| scenario | variables | AUROC | AUPRC | sensitivity | specificity | precision | F1 | Brier | utility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {len(row['variables'])} | {_num(row['auroc'])} | {_num(row['auprc'])} | "
            f"{_num(row['sensitivity'])} | {_num(row['specificity'])} | {_num(row['precision'])} | {_num(row['f1'])} | "
            f"{_num(row['brier'])} | {_num(row['utility'])} |"
        )
    return lines


def _hospital_transfer(phase4: dict) -> list[str]:
    lines = [
        "The transfer experiments train on one hospital and test on the other. "
        "Thresholds were chosen on the source hospital's validation patients. "
        "These models are not the Phase 5 serving candidate. The drop is part of the result.",
        "",
        "| experiment | external patients | hours | AUROC | AUPRC | utility | source validation utility |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, label in (
        ("hospital_A_to_B", "Train A, test B"),
        ("hospital_B_to_A", "Train B, test A"),
    ):
        record = phase4["models"][name]
        metrics = record["test_calibrated"]
        ranking = metrics["ranking"]
        point = metrics["operating_point"]
        lines.append(
            f"| {label} | {record['external_test_patients']} | {ranking['n_hours']} | {_num(ranking['auroc'])} | "
            f"{_num(ranking['auprc'])} | {_num(point['normalized_utility'])} | "
            f"{_num(record['validation_calibrated_operating_point']['normalized_utility'])} |"
        )
    pooled = phase4["models"]["flexible_xgboost"]["test_calibrated"]
    lines.extend(
        [
            "",
            f"For scale, the pooled flexible model on the patient-level test split has AUROC {_num(pooled['ranking']['auroc'])} "
            f"and utility {_num(pooled['operating_point']['normalized_utility'])}. "
            "That test split mixes both hospitals, so it is not an external-hospital result.",
            "",
            "Phase 4 contribution ranks put ICULOS first in both hospital test samples, with HospAdmTime and Unit1 also near the top of both. "
            "Large rank changes were concentrated in missingness indicators. "
            "A plausible reading is that some score comes from site measurement patterns as well as from physiology. "
            "That reading is not proof of the mechanism. It is a reason the pooled number cannot be described as external clinical generalization.",
            "",
            "A later external validation set, from a system that did not contribute these files, is still required.",
        ]
    )
    return lines


def _calibration(phase4: dict) -> list[str]:
    lines = [
        "Calibrators were fit on validation predictions. The test set was not used to choose Platt or isotonic.",
        "",
        "| model | chosen | validation Brier Platt | validation Brier isotonic |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("temporal_last", "flexible_xgboost"):
        report = phase4["models"][name]["calibration"]
        lines.append(
            f"| {name} | {report['chosen']} | {_num(report['validation_brier_platt'])} | {_num(report['validation_brier_isotonic'])} |"
        )
    return lines


def _decision_reason(decision: dict) -> str:
    rows = {row["model_name"]: row for row in decision["candidates"]}
    selected = rows[decision["model_name"]]
    other_name = next(name for name in rows if name != decision["model_name"])
    other = rows[other_name]
    return (
        f"`{decision['model_name']}` is the provisional research serving candidate because its validation "
        f"application score is {_num(selected['application_score'])}. "
        f"Full-input validation utility is {_num(selected['full_utility'])} for the selected model and "
        f"{_num(other['full_utility'])} for `{other_name}`. "
        f"Mean validation utility on random subsets of size 5 and 10 is {_num(selected['sparse_utility'])} "
        f"for the selected model and {_num(other['sparse_utility'])} for `{other_name}`. "
        "The application has to accept arbitrary parameter subsets, so that sparse term is half of the score."
    )


def _num(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{float(value):.6g}"
