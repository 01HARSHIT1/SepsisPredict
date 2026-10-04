"""Fit the phase 2 logistic and gradient-boosted models.

The serving model, its threshold, and its calibrator are chosen on the
validation split. Test rows are scored after that choice. Masking experiments
retrain on the named columns; they do not replace the serving model.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import (
    binary_rates,
    calibration_slope_intercept,
    ranking_metrics,
    reliability_bins,
    threshold_maximising_utility,
    utility_at_threshold,
    warning_lead_hours,
)
from src.inference.artifacts import FEATURE_SPEC_RELATIVE, MANIFEST_RELATIVE, load_json, project_root
from src.inference.contract import CONTRACT_VERSION, PREDICTION_HORIZON
from src.inference.scoring import apply_preprocessor, sigmoid
from src.reporting.phase2 import write_phase2_report
from src.training.dataset import build_phase2_matrices, load_split, load_train_statistics
from src.training.masks import mask_definitions, rank_variables

DATASET_VERSION = "physionet-challenge-2019-1.0.0"
XGB_ROUNDS = 120
XGB_EARLY_STOPPING = 20
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 4,
    "eta": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "max_bin": 128,
    "seed": 2019,
    "nthread": 4,
}
LOGISTIC_EPOCHS = 5
BATCH = 65_536


def feature_schema_version(names: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]
    return f"features-{digest}"


def run_phase2(root: Path | None = None) -> dict:
    root = project_root() if root is None else root
    print("building feature matrices", flush=True)
    build_phase2_matrices(root)
    spec = load_json(root / FEATURE_SPEC_RELATIVE)
    names = [item["name"] for item in spec["features"]]
    checkpoint_path = root / "reports" / "phase2" / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    checkpoint["feature_names"] = names
    checkpoint["feature_schema_version"] = feature_schema_version(names)
    checkpoint["protocol"] = _protocol()
    train = load_split("train", root)
    valid = load_split("validation", root)
    test = load_split("test", root)
    checkpoint["splits"] = {
        name: load_split(name, root)["meta"] for name in ("train", "validation", "test")
    }
    statistics = load_train_statistics(root)
    _fit_full_models(root, checkpoint, checkpoint_path, names, statistics, train, valid, test)
    _select_and_calibrate(root, checkpoint, checkpoint_path, valid, test)
    _fit_masks(root, checkpoint, checkpoint_path, spec, names, statistics, train, valid, test)
    _importance(root, checkpoint, names, valid)
    _write_serving_manifest(root, checkpoint, spec)
    _save_checkpoint(checkpoint_path, checkpoint)
    metrics_path = root / "reports" / "phase2_metrics.json"
    report_path = root / "reports" / "phase2_report.md"
    write_phase2_report(checkpoint, spec, metrics_path, report_path)
    print(f"wrote {report_path}", flush=True)
    return checkpoint


def _fit_full_models(root, checkpoint, checkpoint_path, names, statistics, train, valid, test):
    medians = statistics["medians"].astype(np.float64)
    observed = statistics["median_observations"]
    for balanced in (False, True):
        model_name = "logistic_balanced" if balanced else "logistic_unweighted"
        if model_name in checkpoint["models"]:
            print(f"skip {model_name}", flush=True)
            continue
        print(f"fitting {model_name}", flush=True)
        coefficients, intercept, mean, scale = _fit_logistic(
            train["features"], train["labels"], medians, balanced
        )
        preprocessor = {
            "kind": "median_imputer_standard_scaler",
            "feature_names": names,
            "medians": medians.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "median_reservoir_size": 200_000,
            "median_reservoir_seed": 2019,
            "n_columns_median_exact": int(np.sum(observed <= 200_000)),
            "n_columns_median_reservoir": int(np.sum(observed > 200_000)),
            "columns_with_no_training_observations": [
                names[index] for index, count in enumerate(observed) if int(count) == 0
            ],
        }
        _write_json(root / "artifacts" / "preprocessors" / f"{model_name}.json", preprocessor)
        np.savez(
            root / "artifacts" / "models" / f"{model_name}.npz",
            coefficients=coefficients,
            intercept=np.array([intercept]),
        )
        record = _evaluate_predictor(
            model_name,
            "logistic_regression",
            names,
            lambda block: _logistic_probability(block, preprocessor, coefficients, intercept),
            valid,
            test,
            None,
        )
        record["hyperparameters"] = {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "epochs": LOGISTIC_EPOCHS,
            "class_weight": "balanced" if balanced else "none",
        }
        record["top_coefficients"] = _top_coefficients(names, coefficients)
        checkpoint["models"][model_name] = record
        _save_checkpoint(checkpoint_path, checkpoint)

    n_pos = int(np.asarray(train["labels"]).sum())
    n_neg = int(train["meta"]["n_rows"] - n_pos)
    weight = n_neg / n_pos if n_pos else 1.0
    for model_name, scale_pos_weight in (
        ("xgboost_unweighted", 1.0),
        ("xgboost_scale_pos_weight", weight),
    ):
        if model_name in checkpoint["models"]:
            print(f"skip {model_name}", flush=True)
            continue
        print(f"fitting {model_name}", flush=True)
        booster, best_iteration = _fit_xgboost(
            train["features"],
            train["labels"],
            valid["features"],
            valid["labels"],
            None,
            names,
            scale_pos_weight,
        )
        model_path = root / "artifacts" / "models" / f"{model_name}.json"
        booster.save_model(str(model_path))
        preprocessor = {
            "kind": "native_missing",
            "feature_names": names,
            "note": "Missing values stay missing. No imputer and no scaler is applied.",
        }
        _write_json(root / "artifacts" / "preprocessors" / f"{model_name}.json", preprocessor)
        record = _evaluate_predictor(
            model_name,
            "xgboost",
            names,
            lambda block, booster=booster, best_iteration=best_iteration, feature_names=names: (
                _xgboost_probability(block, booster, feature_names, best_iteration)
            ),
            valid,
            test,
            None,
        )
        record["hyperparameters"] = {
            **XGB_PARAMS,
            "num_boost_round": XGB_ROUNDS,
            "early_stopping_rounds": XGB_EARLY_STOPPING,
            "scale_pos_weight": scale_pos_weight,
            "best_iteration": best_iteration,
        }
        record["gain"] = _gain(booster, names, best_iteration)
        checkpoint["models"][model_name] = record
        _save_checkpoint(checkpoint_path, checkpoint)
        del booster


def _select_and_calibrate(root, checkpoint, checkpoint_path, valid, test):
    if checkpoint.get("serving", {}).get("complete"):
        print("skip calibration", flush=True)
        return
    ranking = []
    for name, record in checkpoint["models"].items():
        ranking.append(
            {
                "model_name": name,
                "validation_utility": record["validation"]["operating_point"]["normalized_utility"],
                "validation_auprc": record["validation"]["ranking"]["auprc"],
            }
        )
    ranking.sort(key=lambda item: (-item["validation_utility"], -item["validation_auprc"], item["model_name"]))
    winner_name = ranking[0]["model_name"]
    print(f"serving candidate {winner_name}", flush=True)
    winner = checkpoint["models"][winner_name]
    probabilities = _saved_probabilities(root, winner_name, winner, valid)
    calibrator, calibration_report = _choose_calibrator(valid["labels"], probabilities)
    calibrated = _apply_saved_calibrator(probabilities, calibrator)
    operating = threshold_maximising_utility(valid["labels"], calibrated, valid["offsets"])
    test_raw = _saved_probabilities(root, winner_name, winner, test)
    test_calibrated = _apply_saved_calibrator(test_raw, calibrator)
    test_metrics = _metric_bundle(
        test["labels"], test_calibrated, test["offsets"], operating["threshold"]
    )
    _write_json(root / "artifacts" / "calibration" / "serving_calibrator.json", calibrator)
    checkpoint["serving"] = {
        "complete": True,
        "model_name": winner_name,
        "model_family": winner["model_family"],
        "selection_ranking": ranking,
        "selection_rule": (
            "Maximum validation normalized utility of uncalibrated probabilities. "
            "Ties use validation AUPRC. The calibrator and the decision threshold "
            "are then fit on validation only."
        ),
        "calibration": calibration_report,
        "validation_calibrated_operating_point": operating,
        "test_calibrated": test_metrics,
        "validation_calibrated_brier_is_optimistic": True,
    }
    _save_checkpoint(checkpoint_path, checkpoint)


def _fit_masks(root, checkpoint, checkpoint_path, spec, names, statistics, train, valid, test):
    ranked = rank_variables(statistics["raw_observation_counts"], int(train["meta"]["n_rows"]))
    checkpoint["variable_ranking"] = ranked
    masks = mask_definitions(spec, ranked)
    checkpoint.setdefault("masks", {})
    family_name, scale_pos_weight = _mask_source(checkpoint, train)
    for mask in masks:
        if mask["name"] in checkpoint["masks"]:
            print(f"skip mask {mask['name']}", flush=True)
            continue
        print(f"fitting mask {mask['name']}", flush=True)
        columns = [names.index(feature) for feature in mask["feature_names"]]
        booster, best_iteration = _fit_xgboost(
            train["features"],
            train["labels"],
            valid["features"],
            valid["labels"],
            columns,
            mask["feature_names"],
            scale_pos_weight,
        )
        model_path = root / "artifacts" / "models" / "masks" / f"{mask['name']}.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_path))
        preprocessor = {
            "kind": "native_missing",
            "feature_names": mask["feature_names"],
            "note": "Mask-specific feature order. Missing values stay missing.",
        }
        preprocessor_path = root / "artifacts" / "preprocessors" / "masks" / f"{mask['name']}.json"
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(preprocessor_path, preprocessor)
        record = _evaluate_predictor(
            mask["name"],
            "xgboost",
            mask["feature_names"],
            lambda block, booster=booster, best_iteration=best_iteration, feature_names=mask["feature_names"]: (
                _xgboost_probability(block, booster, feature_names, best_iteration)
            ),
            valid,
            test,
            columns,
        )
        record.update(
            {
                "variables": mask["variables"],
                "n_variables": mask["n_variables"],
                "n_features": mask["n_features"],
                "feature_names": mask["feature_names"],
                "source_model": family_name,
                "scale_pos_weight": scale_pos_weight,
                "best_iteration": best_iteration,
                "not_a_clinical_reliability_claim": True,
            }
        )
        checkpoint["masks"][mask["name"]] = record
        _save_checkpoint(checkpoint_path, checkpoint)
        del booster


def _importance(root, checkpoint, names, valid):
    if "importance" in checkpoint:
        return
    serving_name = checkpoint["serving"]["model_name"]
    record = checkpoint["models"][serving_name]
    if record["model_family"] == "xgboost":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(root / "artifacts" / "models" / f"{serving_name}.json"))
        best = int(record["hyperparameters"]["best_iteration"])
        rng = np.random.default_rng(2019)
        take = min(4000, valid["features"].shape[0])
        chosen = np.sort(rng.choice(valid["features"].shape[0], size=take, replace=False))
        block = np.array(valid["features"][chosen], dtype=np.float32)
        contributions = _xgboost_contributions(block, booster, names, best)
        mean_abs = np.mean(np.abs(contributions), axis=0)
        order = np.argsort(mean_abs)[::-1][:25]
        checkpoint["importance"] = {
            "method": "mean_absolute_xgboost_pred_contribs",
            "sample_rows": int(take),
            "sample_split": "validation",
            "sample_seed": 2019,
            "features": [
                {"feature": names[int(index)], "mean_absolute_contribution": float(mean_abs[int(index)])}
                for index in order
            ],
        }
    else:
        checkpoint["importance"] = {
            "method": "absolute_logistic_coefficient_on_scaled_features",
            "features": record["top_coefficients"],
        }


def _write_serving_manifest(root, checkpoint, spec):
    serving = checkpoint["serving"]
    name = serving["model_name"]
    record = checkpoint["models"][name]
    family = record["model_family"]
    extension = "npz" if family == "logistic_regression" else "json"
    timestamp = checkpoint["started_at"]
    metadata = {
        "model_name": name,
        "model_version": f"phase2-{timestamp}",
        "model_family": family,
        "training_dataset_version": DATASET_VERSION,
        "feature_schema_version": checkpoint["feature_schema_version"],
        "prediction_target": "SepsisLabel",
        "prediction_horizon": PREDICTION_HORIZON,
        "training_timestamp": timestamp,
        "expected_feature_order": checkpoint["feature_names"],
        "decision_threshold": serving["validation_calibrated_operating_point"]["threshold"],
        "best_iteration": record["hyperparameters"].get("best_iteration"),
        "selection_rule": serving["selection_rule"],
        "contract_version": CONTRACT_VERSION,
        "preprocessor_kind": (
            "median_imputer_standard_scaler" if family == "logistic_regression" else "native_missing"
        ),
    }
    metadata_path = root / "artifacts" / "metadata" / "serving_model.json"
    _write_json(metadata_path, metadata)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": "trained",
        "model_version": metadata["model_version"],
        "model_family": family,
        "model_name": name,
        "prediction_horizon": PREDICTION_HORIZON,
        "feature_spec_path": FEATURE_SPEC_RELATIVE.as_posix(),
        "preprocessor_path": f"artifacts/preprocessors/{name}.json",
        "model_path": f"artifacts/models/{name}.{extension}",
        "calibration_path": "artifacts/calibration/serving_calibrator.json",
        "metadata_path": "artifacts/metadata/serving_model.json",
        "feature_schema_version": checkpoint["feature_schema_version"],
        "decision_threshold": metadata["decision_threshold"],
        "notes": (
            "FastAPI loads these paths. Feature order comes from serving_model.json. "
            "The frontend does not preprocess."
        ),
    }
    _write_json(root / MANIFEST_RELATIVE, manifest)
    spec_path = root / FEATURE_SPEC_RELATIVE
    spec["model_version"] = metadata["model_version"]
    spec["feature_schema_version"] = checkpoint["feature_schema_version"]
    spec["status"] = "feature_definition_with_trained_model"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def _fit_logistic(features, labels, medians, balanced: bool):
    scaler = StandardScaler()
    n_rows = features.shape[0]
    for start in range(0, n_rows, BATCH):
        scaler.partial_fit(_impute(features[start : start + BATCH], medians))
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        random_state=2019,
        learning_rate="optimal",
        tol=None,
        shuffle=False,
        max_iter=1,
    )
    n_pos = int(np.asarray(labels).sum())
    pos_weight = ((n_rows - n_pos) / n_pos) if balanced and n_pos else 1.0
    rng = np.random.default_rng(2019)
    started = False
    for epoch in range(LOGISTIC_EPOCHS):
        print(f"logistic epoch {epoch + 1}/{LOGISTIC_EPOCHS}", flush=True)
        starts = np.arange(0, n_rows, BATCH)
        rng.shuffle(starts)
        for start in starts:
            stop = min(int(start) + BATCH, n_rows)
            transformed = _scale(_impute(features[start:stop], medians), mean, scale)
            y = np.asarray(labels[start:stop], dtype=np.int8)
            weight = np.where(y == 1, pos_weight, 1.0) if balanced else None
            if not started:
                classifier.partial_fit(transformed, y, classes=np.array([0, 1]), sample_weight=weight)
                started = True
            else:
                classifier.partial_fit(transformed, y, sample_weight=weight)
    return classifier.coef_.ravel().astype(np.float64), float(classifier.intercept_[0]), mean, scale


def _fit_xgboost(train_x, train_y, valid_x, valid_y, columns, feature_names, scale_pos_weight):
    import xgboost as xgb

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = float(scale_pos_weight)
    train_matrix = xgb.QuantileDMatrix(
        _BatchIter(train_x, train_y, columns, feature_names).iterator,
        max_bin=128,
    )
    valid_matrix = xgb.QuantileDMatrix(
        _BatchIter(valid_x, valid_y, columns, feature_names).iterator,
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
    best = booster.best_iteration
    if best is None:
        best = XGB_ROUNDS - 1
    return booster, int(best)


class _BatchIter:
    """Yield memmap batches to ``xgboost.QuantileDMatrix``."""

    def __init__(self, features, labels, columns, feature_names):
        import xgboost as xgb

        class _Iterator(xgb.DataIter):
            def __init__(self):
                self.cursor = 0
                super().__init__(release_data=True)

            def reset(self):
                self.cursor = 0

            def next(self, input_data):
                if self.cursor >= features.shape[0]:
                    return False
                stop = min(self.cursor + BATCH, features.shape[0])
                block = np.array(features[self.cursor:stop], dtype=np.float32)
                if columns is not None:
                    block = np.ascontiguousarray(block[:, columns])
                label = np.array(labels[self.cursor:stop], dtype=np.float32)
                input_data(data=block, label=label, feature_names=feature_names)
                self.cursor = stop
                return True

        self._iterator = _Iterator()

    def __iter__(self):
        return self

    @property
    def iterator(self):
        return self._iterator


def _evaluate_predictor(model_name, family, feature_names, predictor, valid, test, columns):
    print(f"evaluating {model_name}", flush=True)
    validation_p = _predict_rows(valid["features"], columns, predictor)
    test_p = _predict_rows(test["features"], columns, predictor)
    validation_choice = threshold_maximising_utility(valid["labels"], validation_p, valid["offsets"])
    return {
        "model_name": model_name,
        "model_family": family,
        "feature_names": feature_names,
        "validation": _metric_bundle(
            valid["labels"], validation_p, valid["offsets"], validation_choice["threshold"]
        ),
        "test_at_validation_threshold": _metric_bundle(
            test["labels"], test_p, test["offsets"], validation_choice["threshold"]
        ),
        "threshold_selected_on": "validation_uncalibrated_utility",
    }


def _metric_bundle(labels, probabilities, offsets, threshold):
    return {
        "ranking": ranking_metrics(labels, probabilities),
        "calibration_line": calibration_slope_intercept(labels, probabilities),
        "reliability": reliability_bins(labels, probabilities),
        "operating_point": {
            **binary_rates(labels, probabilities, threshold),
            "normalized_utility": utility_at_threshold(labels, probabilities, offsets, threshold),
        },
        "warning_lead": warning_lead_hours(labels, probabilities, offsets, threshold),
    }


def _predict_rows(features, columns, predictor) -> np.ndarray:
    out = np.empty(features.shape[0], dtype=np.float64)
    for start in range(0, features.shape[0], BATCH):
        stop = min(start + BATCH, features.shape[0])
        block = np.array(features[start:stop], dtype=np.float32)
        if columns is not None:
            block = block[:, columns]
        out[start:stop] = predictor(block)
    return out


def _logistic_probability(block, preprocessor, coefficients, intercept):
    _transformed, inputs = apply_preprocessor(block, preprocessor)
    margin = inputs @ coefficients + intercept
    return sigmoid(margin)


def _xgboost_probability(block, booster, feature_names, best_iteration):
    import xgboost as xgb

    matrix = xgb.DMatrix(np.asarray(block, dtype=np.float32), feature_names=feature_names, missing=np.nan)
    return booster.predict(matrix, iteration_range=(0, best_iteration + 1))


def _xgboost_contributions(block, booster, feature_names, best_iteration):
    import xgboost as xgb

    matrix = xgb.DMatrix(np.asarray(block, dtype=np.float32), feature_names=feature_names, missing=np.nan)
    contrib = booster.predict(
        matrix, pred_contribs=True, iteration_range=(0, best_iteration + 1)
    )
    return contrib[:, :-1]


def _gain(booster, names, best_iteration):
    score = booster.get_score(importance_type="gain")
    rows = []
    for index, name in enumerate(names):
        value = float(score.get(f"f{index}", score.get(name, 0.0)))
        rows.append({"feature": name, "gain": value})
    rows.sort(key=lambda item: -item["gain"])
    return {"best_iteration": best_iteration, "top": rows[:25]}


def _top_coefficients(names, coefficients):
    order = np.argsort(np.abs(coefficients))[::-1][:25]
    return [
        {"feature": names[int(index)], "coefficient": float(coefficients[int(index)])}
        for index in order
    ]


def _choose_calibrator(labels, probabilities):
    from sklearn.metrics import brier_score_loss

    y = np.asarray(labels).astype(int)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1, 1)
    platt = LogisticRegression(C=1e12, solver="lbfgs", max_iter=200)
    platt.fit(p, y)
    platt_p = platt.predict_proba(p)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p.ravel(), y)
    isotonic_p = isotonic.predict(p.ravel())
    platt_brier = float(brier_score_loss(y, platt_p))
    isotonic_brier = float(brier_score_loss(y, isotonic_p))
    if isotonic_brier < platt_brier:
        chosen = {
            "kind": "isotonic",
            "thresholds": np.asarray(isotonic.X_thresholds_, dtype=np.float64).tolist(),
            "values": np.asarray(isotonic.y_thresholds_, dtype=np.float64).tolist(),
        }
        kind = "isotonic"
    else:
        chosen = {
            "kind": "platt",
            "coefficient": float(platt.coef_[0, 0]),
            "intercept": float(platt.intercept_[0]),
        }
        kind = "platt"
    return chosen, {
        "chosen": kind,
        "validation_brier_platt": platt_brier,
        "validation_brier_isotonic": isotonic_brier,
        "selection": "lower validation Brier; that Brier is fit and scored on the same rows",
    }


def _apply_saved_calibrator(probabilities, calibrator):
    from src.inference.scoring import apply_calibrator

    return apply_calibrator(np.asarray(probabilities, dtype=np.float64), calibrator)


def _saved_probabilities(root, model_name, record, split):
    if record["model_family"] == "logistic_regression":
        preprocessor = load_json(root / "artifacts" / "preprocessors" / f"{model_name}.json")
        arrays = np.load(root / "artifacts" / "models" / f"{model_name}.npz")
        coefficients = np.asarray(arrays["coefficients"], dtype=np.float64)
        intercept = float(arrays["intercept"][0])
        return _predict_rows(
            split["features"],
            None,
            lambda block: _logistic_probability(block, preprocessor, coefficients, intercept),
        )
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(root / "artifacts" / "models" / f"{model_name}.json"))
    names = load_json(root / "artifacts" / "preprocessors" / f"{model_name}.json")["feature_names"]
    best_iteration = int(record["hyperparameters"]["best_iteration"])
    return _predict_rows(
        split["features"],
        None,
        lambda block: _xgboost_probability(block, booster, names, best_iteration),
    )


def _mask_source(checkpoint, train):
    xgb_models = [
        (name, record)
        for name, record in checkpoint["models"].items()
        if record["model_family"] == "xgboost"
    ]
    xgb_models.sort(
        key=lambda item: (
            -item[1]["validation"]["operating_point"]["normalized_utility"],
            -item[1]["validation"]["ranking"]["auprc"],
        )
    )
    name = xgb_models[0][0]
    return name, float(xgb_models[0][1]["hyperparameters"]["scale_pos_weight"])


def _impute(block, medians):
    values = np.array(block, dtype=np.float64, copy=True)
    missing = ~np.isfinite(values)
    if missing.any():
        rows, columns = np.where(missing)
        values[rows, columns] = medians[columns]
    return values


def _scale(values, mean, scale):
    return (values - mean) / scale


def _protocol():
    return {
        "split_seed": 2019,
        "target": "SepsisLabel",
        "serving_selection": "validation normalized utility, then validation AUPRC",
        "calibration_selection": "lower validation Brier of Platt versus isotonic",
        "threshold_selection": "validation normalized utility after calibration for the serving model",
        "masking": (
            "Retrain XGBoost on each variable subset. Count masks use the training-set "
            "raw observation rate, with official column order as the tie break. "
            "These runs are not used to choose the serving model."
        ),
        "xgboost_rounds": XGB_ROUNDS,
        "logistic_epochs": LOGISTIC_EPOCHS,
        "gru": "not trained in this phase",
    }


def _load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"models": {}}


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(checkpoint), indent=2), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")


def _sanitize(value):
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def main() -> None:
    run_phase2()


if __name__ == "__main__":
    main()
