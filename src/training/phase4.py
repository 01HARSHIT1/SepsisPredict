"""Phase 4 experiments.

Phase 2 files are read and copied. They are not replaced. New boosters go
under artifacts/models/phase4_* and artifacts/models/phase2_xgboost/.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from src.data.schema import LABS, STATIC, TIME_CONTEXT, VITALS
from src.evaluation.alarms import false_alarm_summary, lead_time_summary
from src.evaluation.metrics import ranking_metrics, threshold_maximising_utility
from src.inference.artifacts import FEATURE_SPEC_RELATIVE, MANIFEST_RELATIVE, load_json, project_root
from src.inference.contract import CONTRACT_VERSION, PREDICTION_HORIZON
from src.inference.features import INPUT_COLUMNS
from src.models.gru_inputs import gru_feature_names
from src.models.sequences import window_coverage
from src.reporting.phase4 import write_phase4_report
from src.training.dataset import load_split, load_train_statistics
from src.training.flexible_mask import (
    EVAL_SEED,
    TRAIN_SEED,
    UNIFORM_DROP_PROBABILITY,
    apply_drop_mask,
    evaluation_independent_drop,
    keep_variables_drop,
    layout_from_spec,
    protocol_description,
    scenario_drop_rows,
    training_drop_mask,
)
from src.training.gru_train import predict_gru, train_gru
from src.training.hospital_split import assert_patient_disjoint, patient_ids_for
from src.training.masks import SELECTED_LABS, rank_variables
from src.training.train import (
    BATCH,
    DATASET_VERSION,
    XGB_EARLY_STOPPING,
    XGB_PARAMS,
    XGB_ROUNDS,
    _apply_saved_calibrator,
    _choose_calibrator,
    _metric_bundle,
    _sanitize,
    _xgboost_contributions,
    _xgboost_probability,
    feature_schema_version,
)

PHASE2_MODEL = "xgboost_scale_pos_weight"
STRUCTURED_EVALS = (
    "drop_vitals",
    "drop_labs",
    "drop_selected_labs",
    "only_rapid_vitals",
    "only_sparse_labs",
    "static_and_time_only",
)
INFORMATION_LEVELS = (40, 30, 20, 15, 10, 5, 3, 2, 1)


def run_phase4(root: Path | None = None) -> dict:
    root = project_root() if root is None else root
    _preserve_phase2(root)
    phase2_manifest = load_json(root / MANIFEST_RELATIVE)
    checkpoint_path = root / "reports" / "phase4" / "checkpoint.json"
    checkpoint = _load(checkpoint_path)
    checkpoint.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    spec = load_json(root / FEATURE_SPEC_RELATIVE)
    names = [item["name"] for item in spec["features"]]
    checkpoint["feature_schema_version"] = feature_schema_version(names)
    checkpoint["protocol"] = _protocol()
    train = load_split("train", root)
    valid = load_split("validation", root)
    test = load_split("test", root)
    statistics = load_train_statistics(root)
    ranked_rows = rank_variables(statistics["raw_observation_counts"], int(train["meta"]["n_rows"]))
    ranked = [row["variable"] for row in ranked_rows]
    rates = np.asarray([row["observation_rate"] for row in ranked_rows], dtype=np.float64)
    rate_by_name = {row["variable"]: row["observation_rate"] for row in ranked_rows}
    ordered_rates = np.asarray([rate_by_name[name] for name in INPUT_COLUMNS], dtype=np.float64)
    checkpoint["availability_ranking"] = ranked_rows
    checkpoint["masking_protocol"] = protocol_description(ordered_rates.tolist())
    layout = layout_from_spec(spec)
    assignments = _assignments(root)
    hospitals = {split_name: _hospital_map(assignments) for split_name in ("train", "validation", "test")}
    _assert_cross_hospital_ids(assignments)

    _train_standard_models(
        root, checkpoint, checkpoint_path, spec, names, layout, ordered_rates, ranked, train, valid, test, assignments
    )
    _train_grus(root, checkpoint, checkpoint_path, spec, names, train, valid, test)
    _flexible_evaluations(
        root, checkpoint, checkpoint_path, names, layout, ordered_rates, ranked, valid, test
    )
    _importance(root, checkpoint, checkpoint_path, names, test, assignments)
    _alarms_and_thresholds(root, checkpoint, checkpoint_path, valid, test, assignments)
    _recommend(root, checkpoint, spec, names)
    _save(checkpoint_path, checkpoint)
    _assert_phase2_manifest_unchanged(root, phase2_manifest)
    write_phase4_report(root, checkpoint, spec)
    print(f"wrote {root / 'reports' / 'phase4_report.md'}", flush=True)
    return checkpoint


def _train_standard_models(root, checkpoint, checkpoint_path, spec, names, layout, rates, ranked, train, valid, test, assignments):
    jobs = [("flexible_xgboost", None, None, "flexible")]
    jobs.extend(_hospital_jobs(assignments))
    jobs.extend(_ablation_jobs(spec, names))
    for name, columns, feature_names, kind in jobs:
        if checkpoint.get("models", {}).get(name, {}).get("complete"):
            print(f"skip {name}", flush=True)
            continue
        print(f"fitting {name}", flush=True)
        started = time.perf_counter()
        train_features, train_labels, train_range = train["features"], train["labels"], None
        valid_features, valid_labels, valid_range = valid["features"], valid["labels"], None
        mask_fn = None
        if kind == "flexible":
            mask_fn = lambda block, row_indexes, layout=layout, rates=rates, ranked=ranked: apply_drop_mask(
                block, training_drop_mask(row_indexes, rates, ranked, TRAIN_SEED), layout
            )
            feature_names = names
            columns = None
        elif kind == "hospital_A_to_B":
            train_range = _one_range(train, _ids(assignments, "A", "train"))
            valid_range = _one_range(valid, _ids(assignments, "A", "validation"))
            feature_names = names
            columns = None
        elif kind == "hospital_B_to_A":
            train_range = _one_range(train, _ids(assignments, "B", "train"))
            valid_range = _one_range(valid, _ids(assignments, "B", "validation"))
            feature_names = names
            columns = None
        booster, best_iteration, weight = _fit_xgb(
            train_features,
            train_labels,
            train_range,
            valid_features,
            valid_labels,
            valid_range,
            columns,
            feature_names,
            mask_fn,
        )
        model_dir = root / "artifacts" / "models" / f"phase4_{name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_dir / "model.json"))
        preprocessor = {
            "kind": "native_missing",
            "feature_names": feature_names,
            "note": "Missing values stay missing. Artificial masks blank a variable and set its missing indicator to 1.",
        }
        preprocessor_path = root / "artifacts" / "preprocessors" / f"phase4_{name}.json"
        _write(preprocessor_path, preprocessor)
        record = _score_saved_model(
            name,
            booster,
            best_iteration,
            feature_names,
            names,
            train,
            valid,
            test,
            assignments,
            kind,
            None,
        )
        record.update(
            {
                "complete": True,
                "hyperparameters": {
                    **XGB_PARAMS,
                    "num_boost_round": XGB_ROUNDS,
                    "early_stopping_rounds": XGB_EARLY_STOPPING,
                    "scale_pos_weight": weight,
                    "best_iteration": best_iteration,
                },
                "runtime_seconds": time.perf_counter() - started,
                "process_peak_rss_bytes": int(psutil.Process().memory_info().rss),
                "model_path": f"artifacts/models/phase4_{name}/model.json",
                "preprocessor_path": preprocessor_path.relative_to(root).as_posix(),
                "feature_names": feature_names,
            }
        )
        checkpoint.setdefault("models", {})[name] = record
        _save(checkpoint_path, checkpoint)
        del booster
        gc.collect()


def _hospital_jobs(assignments):
    return [
        ("hospital_A_to_B", None, None, "hospital_A_to_B"),
        ("hospital_B_to_A", None, None, "hospital_B_to_A"),
    ]


def _ablation_jobs(spec, names):
    admin = {"Unit1", "Unit2", "HospAdmTime"}
    jobs = []
    specs = {
        "ablation_no_admin": [name for name in INPUT_COLUMNS if name not in admin],
        "ablation_physiology_age": list(VITALS) + list(LABS) + ["Age"],
        "ablation_physiology": list(VITALS) + list(LABS),
        "ablation_no_iculos": [name for name in INPUT_COLUMNS if name != "ICULOS"],
        "temporal_current": None,
        "temporal_missing": None,
        "temporal_last": None,
    }
    for name, variables in specs.items():
        if name.startswith("temporal"):
            feature_names = _temporal_names(spec, name)
        else:
            feature_names = [item["name"] for item in spec["features"] if item["variable"] in set(variables)]
        columns = [names.index(feature) for feature in feature_names]
        jobs.append((name, columns, feature_names, "columns"))
    return jobs


def _temporal_names(spec: dict, level: str) -> list[str]:
    names = []
    for item in spec["features"]:
        variable = item["variable"]
        feature = item["name"]
        suffix = "" if feature == variable else feature[len(variable) + 1 :]
        empirical = item["empirical_class"]
        if level == "temporal_current" and (suffix in {"raw", "value"} or empirical == "time_context"):
            names.append(feature)
        elif level == "temporal_missing" and (
            suffix in {"raw", "value", "missing"} or empirical == "time_context"
        ):
            names.append(feature)
        elif level == "temporal_last" and (
            suffix in {"raw", "value", "missing"}
            or empirical == "time_context"
            or (empirical in {"intermittent", "sparse"} and suffix in {"last", "time_since_measurement"})
        ):
            names.append(feature)
    return names


def _train_grus(root, checkpoint, checkpoint_path, spec, names, train, valid, test):
    feature_names = gru_feature_names(spec)
    columns = [names.index(name) for name in feature_names]
    lengths = np.diff(train["offsets"])
    checkpoint["gru_feature_count"] = len(feature_names)
    checkpoint["gru_feature_names"] = feature_names
    for window in (6, 12, 24):
        key = str(window)
        checkpoint.setdefault("gru", {})
        if checkpoint["gru"].get(key, {}).get("complete"):
            print(f"skip gru {window}", flush=True)
            continue
        print(f"fitting gru window {window}", flush=True)
        started = time.perf_counter()
        coverage = window_coverage(lengths, window)
        trained = train_gru(
            train["features"],
            train["labels"],
            train["offsets"],
            valid["features"],
            valid["labels"],
            valid["offsets"],
            columns,
            feature_names,
            window,
        )
        import torch

        model_dir = root / "artifacts" / "models" / "phase4_gru"
        model_dir.mkdir(parents=True, exist_ok=True)
        state_path = model_dir / f"window_{window}.pt"
        torch.save(trained["model"].state_dict(), state_path)
        config = {
            "kind": "gru_window",
            "window": window,
            "hidden_size": trained["hidden_size"],
            "feature_names": feature_names,
            "scaler_mean": [trained["scaler"][name][0] for name in feature_names],
            "scaler_scale": [trained["scaler"][name][1] for name in feature_names],
            "state_dict_path": state_path.relative_to(root).as_posix(),
            "fill_rule": (
                "Non-finite measurement inputs are scaled with training mean and scale, then filled with 0. "
                "Missing indicators remain 0 or 1. A filled 0 is not a measured zero."
            ),
        }
        config_path = root / "artifacts" / "preprocessors" / f"phase4_gru_window_{window}.json"
        _write(config_path, config)
        valid_p = predict_gru(
            trained["model"],
            valid["features"],
            valid["offsets"],
            columns,
            feature_names,
            trained["scaler"],
            trained["pad"],
            window,
        )
        test_p = predict_gru(
            trained["model"],
            test["features"],
            test["offsets"],
            columns,
            feature_names,
            trained["scaler"],
            trained["pad"],
            window,
        )
        record = _calibrated_record(
            f"gru_window_{window}", valid["labels"], valid_p, valid["offsets"], test["labels"], test_p, test["offsets"]
        )
        record.update(
            {
                "complete": True,
                "window": window,
                "coverage": coverage,
                "history": trained["history"],
                "hidden_size": trained["hidden_size"],
                "n_features": len(feature_names),
                "runtime_seconds": time.perf_counter() - started,
                "process_rss_bytes": int(psutil.Process().memory_info().rss),
                "model_path": state_path.relative_to(root).as_posix(),
                "preprocessor_path": config_path.relative_to(root).as_posix(),
                "feature_names": feature_names,
            }
        )
        checkpoint["gru"][key] = record
        _save(checkpoint_path, checkpoint)
        del trained
        gc.collect()


def _flexible_evaluations(root, checkpoint, checkpoint_path, names, layout, rates, ranked, valid, test):
    if checkpoint.get("flexible_evaluations_complete"):
        return
    model = checkpoint["models"]["flexible_xgboost"]
    booster, best, feature_names = _load_phase4_booster(root, "flexible_xgboost", model)
    threshold = model["validation_calibrated_operating_point"]["threshold"]
    calibrator = load_json(root / model["calibration_path"])
    curve = []
    for count in INFORMATION_LEVELS:
        keep = set(ranked[:count])
        print(f"flexible availability-ranked {count}", flush=True)
        probabilities = _predict_matrix(test["features"], names, feature_names, booster, best, layout, keep_variables_drop(test["features"].shape[0], keep))
        curve.append(
            {
                "label": "availability-ranked feature mask",
                "n_variables": count,
                "variables": ranked[:count],
                **_masked_metrics(test, probabilities, calibrator, threshold),
            }
        )
    one_variable = []
    for variable in (
        "HR",
        "Temp",
        "MAP",
        "Resp",
        "O2Sat",
        "SBP",
        "DBP",
        "Lactate",
        "Creatinine",
        "WBC",
        "Platelets",
        "Bilirubin_total",
        "Glucose",
        "Age",
        "Gender",
        "ICULOS",
    ):
        print(f"flexible one variable {variable}", flush=True)
        probabilities = _predict_matrix(
            test["features"],
            names,
            feature_names,
            booster,
            best,
            layout,
            keep_variables_drop(test["features"].shape[0], {variable}),
        )
        one_variable.append(
            {
                "variable": variable,
                "clinical_reliability_claim": False,
                **_masked_metrics(test, probabilities, calibrator, threshold),
            }
        )
    structured = {}
    for scenario in STRUCTURED_EVALS:
        print(f"flexible structured {scenario}", flush=True)
        drop = scenario_drop_rows(test["features"].shape[0], scenario, ranked)
        probabilities = _predict_matrix(test["features"], names, feature_names, booster, best, layout, drop)
        structured[scenario] = _masked_metrics(test, probabilities, calibrator, threshold)
    row_indexes = np.arange(test["features"].shape[0])
    random_drop = evaluation_independent_drop(row_indexes, np.full(len(INPUT_COLUMNS), UNIFORM_DROP_PROBABILITY), EVAL_SEED)
    weighted_drop = evaluation_independent_drop(row_indexes, 1.0 - rates, EVAL_SEED + 1)
    protocols = {
        "uniform_drop_0.5": _masked_metrics(
            test,
            _predict_matrix(test["features"], names, feature_names, booster, best, layout, random_drop),
            calibrator,
            threshold,
        ),
        "availability_weighted": _masked_metrics(
            test,
            _predict_matrix(test["features"], names, feature_names, booster, best, layout, weighted_drop),
            calibrator,
            threshold,
        ),
    }
    checkpoint["flexible_curve"] = curve
    checkpoint["flexible_one_variable"] = one_variable
    checkpoint["flexible_structured"] = structured
    checkpoint["flexible_protocols"] = protocols
    checkpoint["flexible_evaluations_complete"] = True
    _save(checkpoint_path, checkpoint)
    del booster
    gc.collect()


def _importance(root, checkpoint, checkpoint_path, names, test, assignments):
    if checkpoint.get("importance", {}).get("complete"):
        return
    print("feature contribution stability", flush=True)
    phase2 = _load_phase2_booster(root)
    id_map = _hospital_map(assignments)
    rows = {
        "pooled_test": np.arange(test["features"].shape[0]),
        "hospital_A_test": _rows_for_hospital(test, "A", id_map),
        "hospital_B_test": _rows_for_hospital(test, "B", id_map),
    }
    phase2_tables = {
        key: _mean_abs(phase2["booster"], test["features"], indexes, names, phase2["names"], phase2["best"], 2019 + offset)
        for offset, (key, indexes) in enumerate(rows.items())
    }
    clinical = checkpoint["models"]["ablation_physiology_age"]
    clinical_booster, clinical_best, clinical_names = _load_phase4_booster(root, "ablation_physiology_age", clinical)
    clinical_table = _mean_abs(
        clinical_booster,
        test["features"],
        rows["pooled_test"],
        names,
        clinical_names,
        clinical_best,
        2027,
    )
    checkpoint["importance"] = {
        "complete": True,
        "method": "mean_absolute_xgboost_pred_contribs",
        "sample_rows": 3000,
        "sample_split": "test",
        "phase2_model": PHASE2_MODEL,
        "phase2": phase2_tables,
        "physiology_age_ablation": clinical_table,
        "stability": _stability(phase2_tables["hospital_A_test"], phase2_tables["hospital_B_test"]),
    }
    _save(checkpoint_path, checkpoint)


def _alarms_and_thresholds(root, checkpoint, checkpoint_path, valid, test, assignments):
    if checkpoint.get("analysis_complete"):
        return
    print("alarm and threshold analysis", flush=True)
    phase2 = _load_phase2_booster(root)
    phase2_calibrator = load_json(root / "artifacts" / "calibration" / "serving_calibrator.json")
    phase2_threshold = float(load_json(root / "artifacts" / "metadata" / "serving_model.json")["decision_threshold"])
    phase2_valid = _apply_saved_calibrator(
        _predict_matrix(valid["features"], phase2["names"], phase2["names"], phase2["booster"], phase2["best"], None, None),
        phase2_calibrator,
    )
    phase2_test = _apply_saved_calibrator(
        _predict_matrix(test["features"], phase2["names"], phase2["names"], phase2["booster"], phase2["best"], None, None),
        phase2_calibrator,
    )
    flexible = checkpoint["models"]["flexible_xgboost"]
    flexible_booster, flexible_best, flexible_names = _load_phase4_booster(root, "flexible_xgboost", flexible)
    flexible_calibrator = load_json(root / flexible["calibration_path"])
    flexible_valid = _apply_saved_calibrator(
        _predict_matrix(valid["features"], flexible_names, flexible_names, flexible_booster, flexible_best, None, None),
        flexible_calibrator,
    )
    flexible_test = _apply_saved_calibrator(
        _predict_matrix(test["features"], flexible_names, flexible_names, flexible_booster, flexible_best, None, None),
        flexible_calibrator,
    )
    checkpoint["alarms"] = {
        "phase2_served_model": _alarm_pair(test, phase2_test, phase2_threshold),
        "flexible_xgboost": _alarm_pair(test, flexible_test, flexible["validation_calibrated_operating_point"]["threshold"]),
    }
    checkpoint["threshold_analysis"] = {
        "phase2_served_model": _threshold_table(valid["labels"], phase2_valid, valid["offsets"]),
        "flexible_xgboost": _threshold_table(valid["labels"], flexible_valid, valid["offsets"]),
        "selection_rule": "Maximum validation normalized utility. Ties keep the lower threshold. Test rows are not used.",
    }
    checkpoint["analysis_complete"] = True
    _save(checkpoint_path, checkpoint)


def _recommend(root, checkpoint, spec, names):
    candidates = []
    phase2_summary = _phase2_summary(root)
    candidates.append(
        {
            "model_name": PHASE2_MODEL,
            "family": "xgboost",
            "flexible_input": False,
            "validation_utility": phase2_summary["validation_calibrated_utility"],
            "validation_auprc": phase2_summary["validation_auprc"],
            "source": "phase2",
        }
    )
    for name, record in checkpoint.get("models", {}).items():
        if name.startswith("hospital_"):
            continue
        candidates.append(_candidate_row(name, "xgboost", name == "flexible_xgboost", record))
    gru_records = [record for record in checkpoint.get("gru", {}).values() if record.get("complete")]
    if gru_records:
        chosen_gru = max(
            gru_records,
            key=lambda record: (
                record["validation_calibrated_operating_point"]["normalized_utility"],
                record["validation_uncalibrated"]["ranking"]["auprc"],
            ),
        )
        candidates.append(_candidate_row(chosen_gru["model_name"], "gru", False, chosen_gru))
        checkpoint["gru_validation_selection"] = {
            "model_name": chosen_gru["model_name"],
            "window": chosen_gru["window"],
            "rule": "highest pooled validation utility among GRU windows, then validation AUPRC",
        }
    candidates.sort(key=lambda row: (-row["validation_utility"], -row["validation_auprc"], row["model_name"]))
    winner = candidates[0]
    checkpoint["recommendation"] = {
        "label": "provisional research serving candidate",
        "model_name": winner["model_name"],
        "model_family": winner["family"],
        "flexible_input": winner["flexible_input"],
        "selection_rule": (
            "Highest pooled-validation normalized utility after calibration. "
            "Ties use validation AUPRC. Hospital-transfer models are reported separately "
            "and are not in this comparison because their validation population is different."
        ),
        "candidates": candidates,
        "external_validation_still_required": True,
    }
    _write_phase4_manifest(root, checkpoint, spec, names, winner)


def _write_phase4_manifest(root, checkpoint, spec, names, winner):
    timestamp = checkpoint["started_at"]
    if winner["source"] == "phase2":
        manifest_paths = {
            "model_path": "artifacts/models/xgboost_scale_pos_weight.json",
            "preprocessor_path": "artifacts/preprocessors/xgboost_scale_pos_weight.json",
            "calibration_path": "artifacts/calibration/serving_calibrator.json",
            "feature_names": names,
            "threshold": float(load_json(root / "artifacts" / "metadata" / "serving_model.json")["decision_threshold"]),
            "best_iteration": int(
                load_json(root / "reports" / "phase2" / "checkpoint.json")["models"][PHASE2_MODEL]["hyperparameters"][
                    "best_iteration"
                ]
            ),
        }
    else:
        record = _winner_record(checkpoint, winner["model_name"])
        manifest_paths = {
            "model_path": record["model_path"],
            "preprocessor_path": record["preprocessor_path"],
            "calibration_path": record["calibration_path"],
            "feature_names": record["feature_names"],
            "threshold": record["validation_calibrated_operating_point"]["threshold"],
            "best_iteration": record.get("hyperparameters", {}).get("best_iteration"),
        }
    metadata = {
        "model_name": winner["model_name"],
        "model_version": f"phase4-{timestamp}",
        "model_family": winner["family"],
        "role": "provisional_research_serving_candidate",
        "training_dataset_version": DATASET_VERSION,
        "feature_schema_version": checkpoint["feature_schema_version"],
        "prediction_target": "SepsisLabel",
        "prediction_horizon": PREDICTION_HORIZON,
        "training_timestamp": timestamp,
        "expected_feature_order": manifest_paths["feature_names"],
        "decision_threshold": manifest_paths["threshold"],
        "best_iteration": manifest_paths["best_iteration"],
        "flexible_input": winner["flexible_input"],
        "supported_input_variables": list(INPUT_COLUMNS),
        "calibrator": "see calibration_path",
        "contract_version": CONTRACT_VERSION,
        "external_validation_still_required": True,
    }
    metadata_path = root / "artifacts" / "metadata" / "phase4_serving_model.json"
    _write(metadata_path, metadata)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": "trained",
        "role": "provisional_research_serving_candidate",
        "model_version": metadata["model_version"],
        "model_family": winner["family"],
        "model_name": winner["model_name"],
        "prediction_horizon": PREDICTION_HORIZON,
        "feature_spec_path": FEATURE_SPEC_RELATIVE.as_posix(),
        "preprocessor_path": manifest_paths["preprocessor_path"],
        "model_path": manifest_paths["model_path"],
        "calibration_path": manifest_paths["calibration_path"],
        "metadata_path": "artifacts/metadata/phase4_serving_model.json",
        "feature_schema_version": checkpoint["feature_schema_version"],
        "decision_threshold": manifest_paths["threshold"],
        "flexible_input": winner["flexible_input"],
        "supported_input_variables": list(INPUT_COLUMNS),
        "phase2_manifest_path": "artifacts/metadata/phase2_model_manifest.json",
        "notes": (
            "Provisional research serving candidate selected on pooled validation utility. "
            "Phase 2 artifacts and artifacts/metadata/model_manifest.json are unchanged."
        ),
    }
    _write(root / "artifacts" / "metadata" / "phase4_model_manifest.json", manifest)


def _fit_xgb(train_x, train_y, train_range, valid_x, valid_y, valid_range, columns, feature_names, mask_fn):
    import xgboost as xgb

    train_rows = _range_labels(train_y, train_range)
    positive = float(np.asarray(train_rows).sum())
    weight = (len(train_rows) - positive) / positive if positive else 1.0
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = float(weight)
    train_matrix = xgb.QuantileDMatrix(
        _RowIter(train_x, train_y, train_range, columns, feature_names, mask_fn).iterator, max_bin=128
    )
    valid_matrix = xgb.QuantileDMatrix(
        _RowIter(valid_x, valid_y, valid_range, columns, feature_names, None).iterator,
        ref=train_matrix,
        max_bin=128,
    )
    booster = xgb.train(
        params,
        train_matrix,
        num_boost_round=XGB_ROUNDS,
        evals=[(valid_matrix, "validation")],
        early_stopping_rounds=XGB_EARLY_STOPPING,
        verbose_eval=20,
    )
    best = XGB_ROUNDS - 1 if booster.best_iteration is None else int(booster.best_iteration)
    return booster, best, weight


class _RowIter:
    def __init__(self, features, labels, row_range, columns, feature_names, mask_fn):
        import xgboost as xgb

        start, stop = (0, features.shape[0]) if row_range is None else row_range

        class _Iterator(xgb.DataIter):
            def __init__(self):
                self.cursor = start
                super().__init__(release_data=True)

            def reset(self):
                self.cursor = start

            def next(self, input_data):
                if self.cursor >= stop:
                    return False
                end = min(self.cursor + BATCH, stop)
                block = np.array(features[self.cursor:end], dtype=np.float32)
                rows = np.arange(self.cursor, end)
                if mask_fn is not None:
                    block = mask_fn(block, rows)
                if columns is not None:
                    block = np.ascontiguousarray(block[:, columns])
                input_data(
                    data=block,
                    label=np.array(labels[self.cursor:end], dtype=np.float32),
                    feature_names=feature_names,
                )
                self.cursor = end
                return True

        self._iterator = _Iterator()

    def __iter__(self):
        return iter(())

    @property
    def iterator(self):
        return self._iterator


def _fit_xgb_wrap_fix():
    return None


def _score_saved_model(name, booster, best, feature_names, full_names, train, valid, test, assignments, kind, _unused):
    if kind == "hospital_A_to_B":
        return _score_hospital("A", "B", booster, best, feature_names, full_names, train, valid, test, assignments)
    if kind == "hospital_B_to_A":
        return _score_hospital("B", "A", booster, best, feature_names, full_names, train, valid, test, assignments)
    valid_p = _predict_matrix(valid["features"], full_names, feature_names, booster, best, None, None)
    test_p = _predict_matrix(test["features"], full_names, feature_names, booster, best, None, None)
    record = _calibrated_record(name, valid["labels"], valid_p, valid["offsets"], test["labels"], test_p, test["offsets"])
    return record


def _score_hospital(source, external, booster, best, feature_names, full_names, train, valid, test, assignments):
    valid_y, valid_p = _hospital_sequences(valid, _ids(assignments, source, "validation"), full_names, feature_names, booster, best)
    labels, probabilities, offsets = _stack(valid_y, valid_p)
    choice = threshold_maximising_utility(labels, probabilities, offsets)
    calibrator, report = _choose_calibrator(labels, probabilities)
    calibrated = _apply_saved_calibrator(probabilities, calibrator)
    operating = threshold_maximising_utility(labels, calibrated, offsets)
    test_sequences_y = []
    test_sequences_p = []
    for bundle, split_name in ((train, "train"), (valid, "validation"), (test, "test")):
        y_seq, p_seq = _hospital_sequences(
            bundle, _ids(assignments, external, split_name), full_names, feature_names, booster, best
        )
        test_sequences_y.extend(y_seq)
        test_sequences_p.extend(p_seq)
    test_y, test_p, test_offsets = _stack(test_sequences_y, test_sequences_p)
    test_calibrated = _apply_saved_calibrator(test_p, calibrator)
    return {
        "model_name": f"hospital_{source}_to_{external}",
        "source_hospital": source,
        "external_hospital": external,
        "threshold_selected_on": f"hospital {source} validation only",
        "calibration": report,
        "validation_uncalibrated": _metric_bundle(labels, probabilities, offsets, choice["threshold"]),
        "validation_calibrated_operating_point": operating,
        "test_calibrated": _metric_bundle(test_y, test_calibrated, test_offsets, operating["threshold"]),
        "external_test_patients": int(len(test_sequences_y)),
        "external_test_hours": int(test_y.size),
    }


def _calibrated_record(name, valid_y, valid_p, valid_offsets, test_y, test_p, test_offsets):
    choice = threshold_maximising_utility(valid_y, valid_p, valid_offsets)
    calibrator, report = _choose_calibrator(valid_y, valid_p)
    valid_calibrated = _apply_saved_calibrator(valid_p, calibrator)
    operating = threshold_maximising_utility(valid_y, valid_calibrated, valid_offsets)
    test_calibrated = _apply_saved_calibrator(test_p, calibrator)
    calibration_path = Path("artifacts") / "calibration" / f"phase4_{name}.json"
    return {
        "model_name": name,
        "model_family": "gru" if name.startswith("gru") else "xgboost",
        "calibration": report,
        "calibration_path": calibration_path.as_posix(),
        "calibrator_payload": calibrator,
        "validation_uncalibrated": _metric_bundle(valid_y, valid_p, valid_offsets, choice["threshold"]),
        "validation_calibrated_operating_point": operating,
        "test_uncalibrated_ranking": ranking_metrics(test_y, test_p),
        "test_calibrated": _metric_bundle(test_y, test_calibrated, test_offsets, operating["threshold"]),
    }


def _predict_matrix(features, full_names, model_names, booster, best, layout, drop):
    columns = None if model_names == full_names else [full_names.index(name) for name in model_names]
    output = np.empty(features.shape[0], dtype=np.float64)
    for start in range(0, features.shape[0], BATCH):
        stop = min(start + BATCH, features.shape[0])
        block = np.array(features[start:stop], dtype=np.float32)
        if drop is not None:
            block = apply_drop_mask(block, drop[start:stop], layout)
        if columns is not None:
            block = np.ascontiguousarray(block[:, columns])
        output[start:stop] = _xgboost_probability(block, booster, model_names, best)
    return output


def _masked_metrics(split, probabilities, calibrator, threshold):
    calibrated = _apply_saved_calibrator(probabilities, calibrator)
    uncalibrated = ranking_metrics(split["labels"], probabilities)
    operating = _metric_bundle(split["labels"], calibrated, split["offsets"], threshold)
    return {
        "n_hours": uncalibrated["n_hours"],
        "n_positive_hours": uncalibrated["n_positive_hours"],
        "n_negative_hours": uncalibrated["n_hours"] - uncalibrated["n_positive_hours"],
        "uncalibrated_auroc": uncalibrated["auroc"],
        "uncalibrated_auprc": uncalibrated["auprc"],
        "uncalibrated_brier": uncalibrated["brier"],
        "calibrated": operating,
    }


def _mean_abs(booster, features, row_indexes, full_names, model_names, best, seed):
    generator = np.random.default_rng(seed)
    take = min(3000, len(row_indexes))
    chosen = np.sort(generator.choice(row_indexes, size=take, replace=False))
    columns = [full_names.index(name) for name in model_names]
    block = np.array(features[chosen][:, columns], dtype=np.float32)
    contributions = _xgboost_contributions(block, booster, model_names, best)
    mean_abs = np.mean(np.abs(contributions), axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        {"feature": model_names[int(index)], "mean_absolute_contribution": float(mean_abs[int(index)]), "rank": rank}
        for rank, index in enumerate(order, start=1)
    ]


def _stability(left, right):
    rank_left = {row["feature"]: row["rank"] for row in left}
    rank_right = {row["feature"]: row["rank"] for row in right}
    shared = set(rank_left) & set(rank_right)
    deltas = []
    for feature in shared:
        deltas.append(
            {
                "feature": feature,
                "rank_hospital_A": rank_left[feature],
                "rank_hospital_B": rank_right[feature],
                "absolute_rank_change": abs(rank_left[feature] - rank_right[feature]),
            }
        )
    deltas.sort(key=lambda row: -row["absolute_rank_change"])
    admin_markers = ("Unit1", "Unit2", "HospAdmTime", "ICULOS")
    return {
        "largest_rank_changes": deltas[:15],
        "stable_top_20_intersection": sorted(set(_top(left, 20)) & set(_top(right, 20))),
        "administrative_or_time_features": [row for row in deltas if row["feature"].startswith(admin_markers)],
    }


def _top(rows, count):
    return [row["feature"] for row in rows[:count]]


def _threshold_table(labels, probabilities, offsets):
    rows = []
    for threshold in np.round(np.linspace(0.01, 0.99, 50), 2):
        metrics = _metric_bundle(labels, probabilities, offsets, float(threshold))
        point = metrics["operating_point"]
        false_alarms = false_alarm_summary(labels, probabilities, offsets, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": point["sensitivity"],
                "specificity": point["specificity"],
                "precision": point["precision"],
                "f1": point["f1"],
                "normalized_utility": point["normalized_utility"],
                "false_positive_hours_per_nonseptic_patient": false_alarms[
                    "false_positive_hours_per_nonseptic_patient"
                ],
                "fraction_nonseptic_patients_ever_alarmed": false_alarms[
                    "fraction_nonseptic_patients_ever_alarmed"
                ],
            }
        )
    return rows


def _alarm_pair(split, probabilities, threshold):
    return {
        "threshold": float(threshold),
        "lead_time": lead_time_summary(split["labels"], probabilities, split["offsets"], float(threshold)),
        "false_alarms": false_alarm_summary(split["labels"], probabilities, split["offsets"], float(threshold)),
    }


def _load_phase4_booster(root, name, record):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(root / record["model_path"]))
    return booster, int(record["hyperparameters"]["best_iteration"]), record["feature_names"]


def _load_phase2_booster(root):
    import xgboost as xgb

    checkpoint = load_json(root / "reports" / "phase2" / "checkpoint.json")
    record = checkpoint["models"][PHASE2_MODEL]
    booster = xgb.Booster()
    booster.load_model(str(root / "artifacts" / "models" / f"{PHASE2_MODEL}.json"))
    return {
        "booster": booster,
        "best": int(record["hyperparameters"]["best_iteration"]),
        "names": record["feature_names"],
    }


def _winner_record(checkpoint, model_name):
    if model_name in checkpoint.get("models", {}):
        return checkpoint["models"][model_name]
    for record in checkpoint.get("gru", {}).values():
        if record.get("model_name") == model_name:
            return record
    raise KeyError(model_name)


def _candidate_row(name, family, flexible, record):
    return {
        "model_name": name,
        "family": family,
        "flexible_input": flexible,
        "validation_utility": record["validation_calibrated_operating_point"]["normalized_utility"],
        "validation_auprc": record["validation_uncalibrated"]["ranking"]["auprc"],
        "source": "phase4",
    }


def _phase2_summary(root):
    checkpoint = load_json(root / "reports" / "phase2" / "checkpoint.json")
    serving = checkpoint["serving"]
    model = checkpoint["models"][PHASE2_MODEL]
    return {
        "validation_calibrated_utility": serving["validation_calibrated_operating_point"]["normalized_utility"],
        "validation_auprc": model["validation"]["ranking"]["auprc"],
        "test_calibrated": serving["test_calibrated"],
        "validation_uncalibrated_utility": model["validation"]["operating_point"]["normalized_utility"],
    }


def _hospital_sequences(bundle, allowed, full_names, model_names, booster, best):
    grouped: list[tuple[int, int]] = []
    spans: list[tuple[int, int]] = []
    for index, patient_id in enumerate(bundle["patient_ids"]):
        if patient_id not in allowed:
            continue
        start = int(bundle["offsets"][index])
        stop = int(bundle["offsets"][index + 1])
        if grouped and start == grouped[-1][1]:
            grouped[-1] = (grouped[-1][0], stop)
        else:
            grouped.append((start, stop))
        spans.append((start, stop))
    predicted = {}
    for start, stop in grouped:
        block_features = bundle["features"][start:stop]
        predicted[(start, stop)] = _predict_matrix(block_features, full_names, model_names, booster, best, None, None)
    labels = []
    probabilities = []
    for start, stop in spans:
        for range_start, range_stop in grouped:
            if range_start <= start and stop <= range_stop:
                local = predicted[(range_start, range_stop)]
                offset = start - range_start
                labels.append(np.asarray(bundle["labels"][start:stop]))
                probabilities.append(local[offset : offset + (stop - start)])
                break
    return labels, probabilities


def _stack(labels, probabilities):
    stacked_labels = np.concatenate(labels)
    stacked_probabilities = np.concatenate(probabilities)
    offsets = np.zeros(len(labels) + 1, dtype=np.int64)
    for index, sequence in enumerate(labels):
        offsets[index + 1] = offsets[index] + len(sequence)
    return stacked_labels, stacked_probabilities, offsets


def _rows_for_hospital(bundle, hospital, id_map):
    pieces = []
    for index, patient_id in enumerate(bundle["patient_ids"]):
        if id_map[patient_id] == hospital:
            pieces.append(np.arange(int(bundle["offsets"][index]), int(bundle["offsets"][index + 1])))
    return np.concatenate(pieces)


def _one_range(bundle, allowed):
    ranges = []
    for index, patient_id in enumerate(bundle["patient_ids"]):
        if patient_id not in allowed:
            continue
        start = int(bundle["offsets"][index])
        stop = int(bundle["offsets"][index + 1])
        if ranges and start == ranges[-1][1]:
            ranges[-1][1] = stop
        else:
            ranges.append([start, stop])
    if len(ranges) != 1:
        raise ValueError(f"expected one contiguous hospital block, found {len(ranges)}")
    return int(ranges[0][0]), int(ranges[0][1])


def _range_labels(labels, row_range):
    if row_range is None:
        return labels
    return labels[row_range[0] : row_range[1]]


def _ids(assignments, hospital, split):
    return patient_ids_for(assignments, hospital, split)


def _assignments(root):
    import pandas as pd

    return pd.read_csv(root / "data" / "splits" / "patient_splits.csv")


def _hospital_map(assignments):
    return dict(zip(assignments["patient_id"].astype(str), assignments["hospital_system"].astype(str), strict=True))


def _assert_cross_hospital_ids(assignments):
    for source, external in (("A", "B"), ("B", "A")):
        source_ids = _ids(assignments, source, "train") | _ids(assignments, source, "validation")
        external_ids = _ids(assignments, external, None)
        assert_patient_disjoint(source_ids, external_ids)


def _preserve_phase2(root: Path) -> None:
    snapshot = root / "artifacts" / "metadata" / "phase2_model_manifest.json"
    source = root / MANIFEST_RELATIVE
    if not snapshot.exists():
        snapshot.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    destination = root / "artifacts" / "models" / "phase2_xgboost" / "model.json"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "artifacts" / "models" / "xgboost_scale_pos_weight.json", destination)
    for relative in (
        Path("artifacts") / "preprocessors" / "xgboost_scale_pos_weight.json",
        Path("artifacts") / "calibration" / "serving_calibrator.json",
        Path("artifacts") / "metadata" / "serving_model.json",
    ):
        copied = root / "artifacts" / "models" / "phase2_xgboost" / relative.name
        if not copied.exists():
            shutil.copy2(root / relative, copied)


def _assert_phase2_manifest_unchanged(root, original) -> None:
    current = load_json(root / MANIFEST_RELATIVE)
    if current != original:
        raise RuntimeError("phase 2 manifest was modified")


def _protocol():
    return {
        "phase2_artifacts": "read-only",
        "flexible_model": "one XGBoost trained with the documented missingness mixture",
        "separate_model_masks": "phase 2 availability-ranked retrains, cited and not rerun",
        "threshold_rule": "maximum validation normalized utility; lower threshold wins ties",
        "hospital_rule": "early stopping and threshold use the source hospital validation split only",
        "gru_windows": [6, 12, 24],
        "dataset_version": DATASET_VERSION,
    }


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"models": {}}


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(_without_payloads(payload)), indent=2), encoding="utf-8")
    for name, record in payload.get("models", {}).items():
        _dump_calibrator(path.parents[1] if False else path, record)
    for record in payload.get("gru", {}).values():
        _dump_calibrator(path, record)


def _dump_calibrator(checkpoint_path: Path, record: dict) -> None:
    payload = record.get("calibrator_payload")
    relative = record.get("calibration_path")
    if not payload or not relative:
        return
    root = project_root()
    _write(root / relative, payload)


def _without_payloads(payload: dict) -> dict:
    copied = json.loads(json.dumps(_sanitize(payload)))
    for record in copied.get("models", {}).values():
        record.pop("calibrator_payload", None)
    for record in copied.get("gru", {}).values():
        record.pop("calibrator_payload", None)
    return copied


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")


def main() -> None:
    run_phase4()


if __name__ == "__main__":
    main()
