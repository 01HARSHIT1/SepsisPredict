"""Load saved artifacts and score feature rows.

The feature matrix is built elsewhere with ``assemble_patient``. This module
applies only the learned transforms stored next to the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.inference.artifacts import load_json


DISCLAIMER = (
    "Research estimate of the PhysioNet 2019 early-warning label. "
    "Not a diagnosis. data_sufficiency is a technical count of vital and laboratory "
    "values at the scored hour, not clinical confidence."
)
EXPLANATION_TOP_K = 8


def load_serving_bundle(manifest: dict, root: Path) -> dict:
    metadata_path = root / manifest["metadata_path"]
    metadata = load_json(metadata_path)
    preprocessor = load_json(root / manifest["preprocessor_path"])
    order = metadata["expected_feature_order"]
    if order != preprocessor["feature_names"]:
        raise ValueError("model metadata and preprocessor do not list the same feature order")
    model_path = root / manifest["model_path"]
    family = manifest["model_family"]
    if family == "logistic_regression":
        model = np.load(model_path)
        estimator = {
            "coefficients": np.asarray(model["coefficients"], dtype=np.float64),
            "intercept": float(np.asarray(model["intercept"]).reshape(-1)[0]),
        }
    elif family == "xgboost":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        estimator = booster
    else:
        raise ValueError(f"unsupported model family: {family}")
    calibrator = None
    if manifest.get("calibration_path"):
        calibrator = load_json(root / manifest["calibration_path"])
    return {
        "metadata": metadata,
        "preprocessor": preprocessor,
        "estimator": estimator,
        "calibrator": calibrator,
        "family": family,
        "best_iteration": metadata.get("best_iteration"),
    }


def score_feature_matrix(
    features: np.ndarray,
    feature_names: list[str],
    bundle: dict,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Return probabilities and per-row model contributions for the served feature order."""
    order = bundle["metadata"]["expected_feature_order"]
    missing = [name for name in order if name not in feature_names]
    if missing:
        raise ValueError(f"feature matrix is missing {missing[:5]}")
    indexes = [feature_names.index(name) for name in order]
    selected = np.asarray(features[:, indexes], dtype=np.float32)
    transformed, contribution_inputs = apply_preprocessor(selected, bundle["preprocessor"])
    raw, contributions = _predict(transformed, contribution_inputs, order, bundle)
    probabilities = apply_calibrator(raw, bundle["calibrator"])
    explanations = [clinical_explanations(row, order) for row in contributions]
    return probabilities, explanations


def apply_preprocessor(selected: np.ndarray, preprocessor: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return model inputs and the values used to form linear contributions.

    Tree models keep missing values. Logistic regression replaces missing
    entries with the training median and then applies the training scaler.
    """
    kind = preprocessor["kind"]
    if kind == "native_missing":
        return selected, selected
    if kind != "median_imputer_standard_scaler":
        raise ValueError(f"unsupported preprocessor kind: {kind}")
    medians = np.asarray(preprocessor["medians"], dtype=np.float64)
    mean = np.asarray(preprocessor["mean"], dtype=np.float64)
    scale = np.asarray(preprocessor["scale"], dtype=np.float64)
    imputed = np.array(selected, dtype=np.float64, copy=True)
    missing = ~np.isfinite(imputed)
    if missing.any():
        row_index, column_index = np.where(missing)
        imputed[row_index, column_index] = medians[column_index]
    scaled = (imputed - mean) / scale
    return scaled.astype(np.float32), scaled


def apply_calibrator(probabilities: np.ndarray, calibrator: dict | None) -> np.ndarray:
    if calibrator is None:
        return probabilities
    kind = calibrator["kind"]
    if kind == "platt":
        logit = probabilities * float(calibrator["coefficient"]) + float(calibrator["intercept"])
        return 1.0 / (1.0 + np.exp(-logit))
    if kind == "isotonic":
        thresholds = np.asarray(calibrator["thresholds"], dtype=np.float64)
        values = np.asarray(calibrator["values"], dtype=np.float64)
        return np.interp(probabilities, thresholds, values)
    raise ValueError(f"unsupported calibrator kind: {kind}")


def sigmoid(margin: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(margin, -50.0, 50.0)))


def _predict(transformed, contribution_inputs, names, bundle):
    if bundle["family"] == "logistic_regression":
        coefficients = bundle["estimator"]["coefficients"]
        intercept = bundle["estimator"]["intercept"]
        margin = contribution_inputs @ coefficients + intercept
        probabilities = sigmoid(margin)
        contributions = contribution_inputs * coefficients
        return probabilities, contributions
    import xgboost as xgb

    matrix = xgb.DMatrix(transformed, feature_names=names, missing=np.nan)
    iteration = bundle["best_iteration"]
    iteration_range = None if iteration is None else (0, int(iteration) + 1)
    probabilities = bundle["estimator"].predict(matrix, iteration_range=iteration_range)
    contrib = bundle["estimator"].predict(matrix, pred_contribs=True, iteration_range=iteration_range)
    return probabilities, contrib[:, :-1]


def clinical_explanations(contributions: np.ndarray, names: list[str]) -> list[dict]:
    """Sum tree contributions that belong to the same clinical variable.

    The sum is the model's own additive contribution in pre-calibration log odds.
    The returned name is the clinical variable, not an engineered column.
    """
    totals: dict[str, float] = {}
    for name, value in zip(names, contributions, strict=True):
        variable = clinical_variable(str(name))
        totals[variable] = totals.get(variable, 0.0) + float(value)
    ranked = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)
    rows = []
    for variable, total in ranked:
        if total == 0.0:
            continue
        rows.append(
            {
                "feature": variable,
                "contribution": float(total),
                "direction": "increases_risk_score" if total > 0.0 else "decreases_risk_score",
                "contribution_space": "log_odds_before_calibration",
            }
        )
        if len(rows) == EXPLANATION_TOP_K:
            break
    return rows


def clinical_variable(feature_name: str) -> str:
    from src.inference.features import INPUT_COLUMNS

    for variable in sorted(INPUT_COLUMNS, key=len, reverse=True):
        if feature_name == variable or feature_name.startswith(variable + "_"):
            return variable
    return feature_name


def _top_explanations(contributions: np.ndarray, names: list[str]) -> list[dict]:
    return clinical_explanations(contributions, names)


