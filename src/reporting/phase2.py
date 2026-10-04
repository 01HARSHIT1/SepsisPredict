"""Write the phase 2 report from the checkpoint. Numbers are not typed by hand."""

from __future__ import annotations

import json
from pathlib import Path


def write_phase2_report(checkpoint: dict, spec: dict, metrics_path: Path, report_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    report_path.write_text(_render(checkpoint, spec), encoding="utf-8")


def _render(checkpoint: dict, spec: dict) -> str:
    lines: list[str] = []
    lines.append("# Phase 2 model report")
    lines.append("")
    lines.append(
        "This file is generated from `reports/phase2_metrics.json`. "
        "`SepsisLabel` is the PhysioNet 2019 early-warning label, equal to 1 from "
        "six hours before the challenge onset for patients who meet that definition. "
        "It is not a contemporaneous diagnosis. A one-variable result below is an "
        "experimental operating characteristic, not evidence of clinical reliability."
    )
    lines.append("")
    lines.append("## 1. Feature matrix definition")
    lines.append("")
    lines.extend(_feature_section(spec))
    lines.append("")
    lines.append("## 2. Preprocessing pipeline")
    lines.append("")
    lines.extend(_pipeline_section(checkpoint))
    lines.append("")
    lines.append("## 3. Logistic regression baseline")
    lines.append("")
    lines.extend(_model_section(checkpoint, "logistic_unweighted"))
    lines.extend(_model_section(checkpoint, "logistic_balanced"))
    lines.append("")
    lines.append("## 4. Gradient-boosted model")
    lines.append("")
    lines.extend(_model_section(checkpoint, "xgboost_unweighted"))
    lines.extend(_model_section(checkpoint, "xgboost_scale_pos_weight"))
    lines.append("")
    lines.append("## 5. Calibration")
    lines.append("")
    lines.extend(_calibration_section(checkpoint))
    lines.append("")
    lines.append("## 6. Feature masking")
    lines.append("")
    lines.extend(_mask_section(checkpoint))
    lines.append("")
    lines.append("## 7. Temporal leakage tests")
    lines.append("")
    lines.append(
        "Hourly features are built by `src.features.assemble.assemble_patient` and "
        "`src.inference.features.build_feature_row`. Both read only rows at or before "
        "the current hour. `tests/test_leakage.py`, `tests/test_inference_contract.py`, "
        "and `tests/test_assemble_matches_scalar.py` check future-edit invariance, "
        "the absence of backward fill for static values, patient-level splits, and "
        "equality of the vectorized and scalar feature rows. Run `python -m pytest`."
    )
    lines.append("")
    lines.append("## 8. Important features")
    lines.append("")
    lines.extend(_importance_section(checkpoint))
    lines.append("")
    lines.append("## 9. Saved artifacts")
    lines.append("")
    lines.extend(_artifact_section(checkpoint))
    lines.append("")
    lines.append("## 10. Inference command")
    lines.append("")
    lines.append("From the project directory, after the artifacts above exist:")
    lines.append("")
    lines.append("```")
    lines.append("python -m src.inference.score --request request.json")
    lines.append("python -m src.inference.score --model-info")
    lines.append("```")
    lines.append("")
    lines.append(
        "`src.inference.service.predict` is the function a later FastAPI route should call. "
        "It loads the manifest, the feature spec, the preprocessor, the model, and the calibrator. "
        "No HTTP route is started in this phase."
    )
    lines.append("")
    lines.append("## 11. Proposed FastAPI request schema")
    lines.append("")
    lines.extend(_request_schema())
    lines.append("")
    lines.append("## 12. Proposed FastAPI response schema")
    lines.append("")
    lines.extend(_response_schema())
    lines.append("")
    lines.append("## What this phase does not contain")
    lines.append("")
    lines.append(
        "No GRU, LSTM, or temporal-attention model is trained. No Next.js page and no "
        "FastAPI route is implemented. `data_sufficiency` is null. Explanations are "
        "model contributions in pre-calibration log-odds space, not prose diagnoses."
    )
    lines.append("")
    return "\n".join(lines)


def _feature_section(spec: dict) -> list[str]:
    grouped: dict[str, list[str]] = {}
    suffixes: dict[str, list[str]] = {}
    for item in spec["features"]:
        grouped.setdefault(item["empirical_class"], [])
        if item["variable"] not in grouped[item["empirical_class"]]:
            grouped[item["empirical_class"]].append(item["variable"])
        suffix = item["name"][len(item["variable"]) + 1 :] if item["name"] != item["variable"] else "(raw time value)"
        suffixes.setdefault(item["empirical_class"], [])
        if suffix not in suffixes[item["empirical_class"]]:
            suffixes[item["empirical_class"]].append(suffix)
    lines = [
        f"Contract version `{spec.get('contract_version')}`. "
        f"Schema version `{spec.get('feature_schema_version')}`. "
        f"{len(spec['features'])} features in `artifacts/feature_definitions/feature_spec.json` order.",
        "",
        "Static variables are Age, Gender, Unit1, Unit2, and HospAdmTime. They have a carried value and a missing indicator, and they have no hourly delta. ICULOS is the submitted time value only. Near-hourly variables keep raw, missing, last, time since measurement, one-hour and three-hour deltas of measured values, and backward 3-hour and 6-hour summaries. Intermittent and sparse variables keep the current raw value separate from the last observed value, the missing indicator, time since measurement, change from the previous measurement, and a 24-hour measurement count. A carried-forward lab is not treated as a fresh measurement: the raw value is missing on hours without a new measurement.",
        "",
    ]
    for empirical_class, variables in grouped.items():
        lines.append(
            f"- `{empirical_class}` ({len(variables)}): {', '.join(variables)}. "
            f"Suffixes: {', '.join(suffixes[empirical_class])}."
        )
    return lines


def _pipeline_section(checkpoint: dict) -> list[str]:
    return [
        "Training and inference both call `assemble_patient`.",
        "",
        "1. Parse the request into a 40-column hourly matrix. Omitted keys and null stay missing. Numeric 0 stays 0.",
        "2. If the request uses `hour`, sort by hour and insert missing rows for gaps. Score only the submitted hours.",
        "3. At hour t, every feature reads measurements from hour 0 through t.",
        "4. Logistic regression then replaces non-finite features with the training reservoir median and applies the training `StandardScaler` mean and scale. Columns with no training observations use median 0. XGBoost does not impute or scale; non-finite values remain missing.",
        "5. The model emits a probability of `SepsisLabel`. The saved calibrator, chosen on validation, maps that probability. Explanations use the pre-calibration log-odds contributions.",
        "6. Feature order is read from `artifacts/metadata/serving_model.json`. It is not hardcoded in the response builder.",
        "",
        f"Protocol: {json.dumps(checkpoint.get('protocol', {}), indent=2)}",
    ]


def _model_section(checkpoint: dict, name: str) -> list[str]:
    record = checkpoint.get("models", {}).get(name)
    if record is None:
        return [f"### {name}", "", "Not produced.", ""]
    lines = [f"### {name}", ""]
    lines.append(f"Hyperparameters: `{json.dumps(record.get('hyperparameters', {}))}`")
    lines.append("")
    for split_name in ("validation", "test_at_validation_threshold"):
        block = record.get(split_name)
        if not block:
            continue
        lines.append(f"{split_name}:")
        lines.append("")
        lines.append(_metrics_table(block))
        lines.append("")
    lines.append(
        "The threshold in `test_at_validation_threshold` was chosen on uncalibrated "
        "validation utility. It is not the serving threshold when a calibrator is applied afterward."
    )
    lines.append("")
    return lines


def _metrics_table(block: dict) -> str:
    ranking = block["ranking"]
    point = block["operating_point"]
    line = block["calibration_line"]
    lead = block["warning_lead"]
    return "\n".join(
        [
            "| metric | value |",
            "| --- | --- |",
            f"| hours | {ranking['n_hours']} |",
            f"| positive hours | {ranking['n_positive_hours']} |",
            f"| positive-hour rate | {_num(ranking['positive_hour_rate'])} |",
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
            f"| median warning lead hours | {_num(lead['median_lead_hours'])} |",
            f"| sepsis patients with onset inside file | {lead['n_onset_inside_file']} |",
            f"| of those, patients with an alarm | {lead['n_with_alarm']} |",
        ]
    )


def _calibration_section(checkpoint: dict) -> list[str]:
    serving = checkpoint.get("serving")
    if not serving:
        return ["Not produced."]
    lines = [
        f"Serving model: `{serving['model_name']}` ({serving['model_family']}).",
        "",
        serving["selection_rule"],
        "",
        "Validation ranking used for selection:",
        "",
        "| model | validation utility | validation AUPRC |",
        "| --- | --- | --- |",
    ]
    for row in serving["selection_ranking"]:
        lines.append(
            f"| {row['model_name']} | {_num(row['validation_utility'])} | {_num(row['validation_auprc'])} |"
        )
    calibration = serving["calibration"]
    lines.extend(
        [
            "",
            f"Chosen calibrator: `{calibration['chosen']}`.",
            f"Validation Brier, Platt: {_num(calibration['validation_brier_platt'])}.",
            f"Validation Brier, isotonic: {_num(calibration['validation_brier_isotonic'])}.",
            calibration["selection"],
            "",
            "Validation operating point after calibration "
            "(threshold selection uses these same validation rows):",
            "",
            f"Threshold `{_num(serving['validation_calibrated_operating_point']['threshold'])}`, "
            f"normalized utility `{_num(serving['validation_calibrated_operating_point']['normalized_utility'])}`.",
            "",
            "Test metrics of the calibrated serving model at that threshold:",
            "",
            _metrics_table(serving["test_calibrated"]),
        ]
    )
    return lines


def _mask_section(checkpoint: dict) -> list[str]:
    ranking = checkpoint.get("variable_ranking", [])
    lines = [
        "Count masks keep the variables with the highest fraction of training hours observed. "
        "Ties follow the official 40-column order. The full-information models above are the baseline. "
        "Each mask below is a separately trained XGBoost model. Its threshold is chosen on its own "
        "validation utility and then applied to test. Mask results are not used to pick the serving model.",
        "",
    ]
    if ranking:
        lines.append("Training observation ranking:")
        lines.append("")
        lines.append("| rank | variable | observed hours | observation rate |")
        lines.append("| --- | --- | --- | --- |")
        for row in ranking:
            lines.append(
                f"| {row['rank']} | {row['variable']} | {row['observed_hours']} | {_num(row['observation_rate'])} |"
            )
        lines.append("")
    masks = checkpoint.get("masks", {})
    if not masks:
        lines.append("Mask models were not produced.")
        return lines
    lines.append(
        "| mask | variables | features | validation AUPRC | validation utility | test AUPRC | test utility | test sensitivity | test precision |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, record in masks.items():
        validation = record["validation"]
        test = record["test_at_validation_threshold"]
        lines.append(
            "| {name} | {n_variables} | {n_features} | {va} | {vu} | {ta} | {tu} | {sens} | {prec} |".format(
                name=name,
                n_variables=record["n_variables"],
                n_features=record["n_features"],
                va=_num(validation["ranking"]["auprc"]),
                vu=_num(validation["operating_point"]["normalized_utility"]),
                ta=_num(test["ranking"]["auprc"]),
                tu=_num(test["operating_point"]["normalized_utility"]),
                sens=_num(test["operating_point"]["sensitivity"]),
                prec=_num(test["operating_point"]["precision"]),
            )
        )
    lines.append("")
    lines.append("Variables in each mask:")
    lines.append("")
    for name, record in masks.items():
        lines.append(f"- `{name}`: {', '.join(record['variables'])}")
    return lines


def _importance_section(checkpoint: dict) -> list[str]:
    lines = []
    for name, record in checkpoint.get("models", {}).items():
        if record.get("top_coefficients"):
            lines.append(f"`{name}` largest absolute scaled coefficients:")
            lines.append("")
            for row in record["top_coefficients"][:10]:
                lines.append(f"- {row['feature']}: {_num(row['coefficient'])}")
            lines.append("")
        gain = record.get("gain", {}).get("top", [])
        if gain:
            lines.append(f"`{name}` highest XGBoost gain:")
            lines.append("")
            for row in gain[:10]:
                lines.append(f"- {row['feature']}: {_num(row['gain'])}")
            lines.append("")
    importance = checkpoint.get("importance")
    if importance:
        lines.append(f"Serving-model sample: `{importance['method']}`.")
        lines.append("")
        for row in importance["features"][:10]:
            value = row.get("mean_absolute_contribution", row.get("coefficient"))
            lines.append(f"- {row['feature']}: {_num(value)}")
    if not lines:
        lines.append("Not produced.")
    return lines


def _artifact_section(checkpoint: dict) -> list[str]:
    serving = checkpoint.get("serving", {})
    name = serving.get("model_name", "(not selected)")
    return [
        f"Serving model name: `{name}`.",
        f"Feature schema version: `{checkpoint.get('feature_schema_version')}`.",
        f"Training dataset version: `physionet-challenge-2019-1.0.0`.",
        f"Training timestamp: `{checkpoint.get('started_at')}`.",
        "",
        "- `artifacts/metadata/model_manifest.json`",
        "- `artifacts/metadata/serving_model.json` (name, version, dataset, schema, target, horizon, timestamp, feature order, threshold)",
        "- `artifacts/feature_definitions/feature_spec.json`",
        "- `artifacts/models/` for the logistic and XGBoost candidates",
        "- `artifacts/models/masks/` for the masking retrains",
        "- `artifacts/preprocessors/` for the median/scaler or the native-missing feature order",
        "- `artifacts/calibration/serving_calibrator.json`",
        "- `data/processed/phase2/` float32 matrices, labels, and patient offsets",
        "- `reports/phase2_metrics.json`",
        "",
        "Split row counts:",
        "",
        "| split | patients | hours | positive hours | sepsis patients |",
        "| --- | --- | --- | --- | --- |",
    ] + [
        f"| {split} | {meta['n_patients']} | {meta['n_rows']} | {meta['n_positive_hours']} | {meta['n_sepsis_patients']} |"
        for split, meta in checkpoint.get("splits", {}).items()
    ]


def _request_schema() -> list[str]:
    return [
        "Mode A, current sparse values. Missing fields stay missing:",
        "",
        "```json",
        '{"HR": 118, "Temp": 38.7, "MAP": 64, "Resp": 25}',
        "```",
        "",
        "Mode B, explicit hours. The service sorts by `hour`, fills internal gaps with missing values, and returns one prediction per submitted hour:",
        "",
        "```json",
        "[",
        '  {"hour": 1, "HR": 88, "MAP": 82},',
        '  {"hour": 2, "HR": 93, "MAP": 78},',
        '  {"hour": 3, "HR": 101, "MAP": 72}',
        "]",
        "```",
        "",
        "The phase 1 body `{\"observations\": [{\"values\": {...}}]}` remains accepted. `SepsisLabel` is rejected.",
    ]


def _response_schema() -> list[str]:
    return [
        "```json",
        "{",
        '  "contract_version": "1",',
        '  "model_name": "string",',
        '  "model_version": "string",',
        '  "model_family": "logistic_regression | xgboost",',
        '  "prediction_target": "SepsisLabel",',
        '  "prediction_horizon": {"name": "official_sepsis_label", "description": "string"},',
        '  "data_sufficiency": null,',
        '  "disclaimer": "string",',
        '  "predictions": [',
        "    {",
        '      "hour": "integer or null",',
        '      "risk_probability": "number from the loaded model",',
        '      "data_available": "integer count of finite inputs at that hour",',
        '      "data_total": 40,',
        '      "data_completeness": "data_available / 40",',
        '      "data_sufficiency": null,',
        '      "historical_hours": "integer length of the timeline through this hour",',
        '      "risk_trend": "increasing | decreasing | unchanged | null",',
        '      "explanations": [',
        "        {",
        '          "feature": "string from the saved feature order",',
        '          "contribution": "number",',
        '          "contribution_space": "log_odds_before_calibration"',
        "        }",
        "      ]",
        "    }",
        "  ]",
        "}",
        "```",
    ]


def _num(value) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6g}"
