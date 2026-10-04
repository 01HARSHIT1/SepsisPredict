"""Compare the two Phase 4 candidates and write the serving manifest.

No new model is trained. Masks are applied at scoring time. The serving
choice is written from validation scores before any test mask is scored.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluation.metrics import binary_rates, ranking_metrics, utility_at_threshold
from src.evaluation.serving_comparison import (
    ONE_VARIABLES,
    random_subsets,
    select_serving_model,
    sparse_utility,
    summarize_subset_metrics,
    user_scenarios,
)
from src.inference.artifacts import FEATURE_SPEC_RELATIVE, load_json, project_root
from src.inference.contract import PREDICTION_HORIZON
from src.inference.features import INPUT_COLUMNS
from src.training.dataset import load_split
from src.training.flexible_mask import keep_variables_drop, layout_from_spec
from src.training.hospital_split import patient_ids_for
from src.training.phase4 import _assignments, _predict_matrix
from src.training.train import DATASET_VERSION, _apply_saved_calibrator, _sanitize, feature_schema_version

CHECKPOINT_RELATIVE = Path("reports") / "phase5" / "checkpoint.json"
MANIFEST_RELATIVE = Path("artifacts") / "metadata" / "phase5_model_manifest.json"
METADATA_RELATIVE = Path("artifacts") / "metadata" / "phase5_serving_model.json"
PHASE4_CHECKPOINT = Path("reports") / "phase4" / "checkpoint.json"
SPARSE_TRAINING_RATE = 0.05


def run_phase5(root: Path | None = None) -> dict:
    root = project_root() if root is None else root
    path = root / CHECKPOINT_RELATIVE
    checkpoint = _load(path)
    checkpoint.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    spec = load_json(root / FEATURE_SPEC_RELATIVE)
    names = [item["name"] for item in spec["features"]]
    layout = layout_from_spec(spec)
    phase4 = _load(root / PHASE4_CHECKPOINT)
    ranking = [row["variable"] for row in phase4["availability_ranking"]]
    training_rates = {row["variable"]: row["observation_rate"] for row in phase4["availability_ranking"]}
    if "subsets" not in checkpoint:
        checkpoint["subsets"] = {str(size): groups for size, groups in random_subsets().items()}
        _save(path, checkpoint)
    scenarios = user_scenarios(ranking)
    checkpoint["scenarios"] = scenarios
    checkpoint["one_variables"] = list(ONE_VARIABLES)
    assignments = _assignments(root)

    _evaluate_split(
        "validation", root, path, checkpoint, spec, names, layout, training_rates, scenarios, assignments, phase4
    )
    if "decision" not in checkpoint:
        checkpoint["decision"] = _decide(checkpoint, phase4)
        _save(path, checkpoint)
        print(f"validation decision {checkpoint['decision']['model_name']}", flush=True)
    _evaluate_split(
        "test", root, path, checkpoint, spec, names, layout, training_rates, scenarios, assignments, phase4
    )
    _write_manifest(root, checkpoint, spec, names, phase4)
    from src.reporting.phase5 import write_phase5_report

    write_phase5_report(root, checkpoint)
    print(f"wrote {root / 'reports' / 'phase5_report.md'}", flush=True)
    return checkpoint


def _evaluate_split(split_name, root, path, checkpoint, spec, names, layout, training_rates, scenarios, assignments, phase4):
    pending = [
        name
        for name in ("temporal_last", "flexible_xgboost")
        if not checkpoint.get(split_name, {}).get(name, {}).get("complete")
    ]
    if not pending:
        return
    print(f"loading {split_name}", flush=True)
    split = load_split(split_name, root)
    observed = _observation_fractions(split["features"], layout)
    checkpoint.setdefault(split_name, {})
    for name in pending:
        print(f"scoring {name} on {split_name}", flush=True)
        stored = phase4["models"][name]["validation_calibrated_operating_point"]["normalized_utility"]
        checkpoint[split_name][name] = _score_model(
            root,
            phase4["models"][name],
            names,
            layout,
            split,
            observed,
            training_rates,
            checkpoint["subsets"],
            scenarios,
            assignments,
            split_name,
            stored if split_name == "validation" else None,
        )
        _save(path, checkpoint)
        print(f"saved {name} {split_name}", flush=True)


def _score_model(root, record, names, layout, split, observed, training_rates, subsets, scenarios, assignments, split_name, stored_utility):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(root / record["model_path"]))
    best = int(record["hyperparameters"]["best_iteration"])
    feature_names = record["feature_names"]
    threshold = float(record["validation_calibrated_operating_point"]["threshold"])
    calibrator = load_json(root / record["calibration_path"])
    labels = split["labels"]
    offsets = split["offsets"]

    def probabilities(keep: set[str] | None) -> np.ndarray:
        drop = None if keep is None else keep_variables_drop(split["features"].shape[0], keep)
        raw = _predict_matrix(split["features"], names, feature_names, booster, best, layout, drop)
        return _apply_saved_calibrator(raw, calibrator)

    def score(keep: set[str] | None, ready: np.ndarray | None = None) -> dict:
        calibrated = probabilities(keep) if ready is None else ready
        return _operating(labels, calibrated, offsets, threshold)

    full_probabilities = probabilities(None)
    full = score(None, full_probabilities)
    if stored_utility is not None and abs(full["utility"] - stored_utility) > 1e-4:
        raise RuntimeError(f"validation utility recomputed {full['utility']} stored {stored_utility}")
    one_variable = []
    for variable in ONE_VARIABLES:
        print(f"  one variable {variable}", flush=True)
        metrics = score({variable})
        rate = float(training_rates[variable])
        one_variable.append(
            {
                "variable": variable,
                "training_observation_rate": rate,
                "split_observation_rate": float(observed[variable]),
                "too_sparse_for_stable_operating_point": rate < SPARSE_TRAINING_RATE,
                "clinical_reliability_claim": False,
                **metrics,
            }
        )
    random_blocks = {}
    for size, groups in subsets.items():
        rows = []
        for index, variables in enumerate(groups, start=1):
            print(f"  random k={size} subset {index}", flush=True)
            if set(variables) == set(INPUT_COLUMNS):
                metrics = full
            else:
                metrics = score(set(variables))
            rows.append({"variables": variables, **metrics})
        random_blocks[str(size)] = {"subsets": rows, **summarize_subset_metrics(rows)}
    scenario_rows = {}
    for scenario, variables in scenarios.items():
        print(f"  scenario {scenario}", flush=True)
        scenario_rows[scenario] = {"variables": variables, **score(set(variables))}
    hospitals = {}
    if split_name == "validation":
        hospitals = {
            hospital: _slice_metrics(
                labels,
                full_probabilities,
                offsets,
                split["patient_ids"],
                patient_ids_for(assignments, hospital, "validation"),
                threshold,
            )
            for hospital in ("A", "B")
        }
    return {
        "complete": True,
        "threshold": threshold,
        "threshold_source": "natural validation inputs for this model",
        "full": full,
        "one_variable": one_variable,
        "random_subsets": random_blocks,
        "scenarios": scenario_rows,
        "hospital_slice": hospitals,
    }


def _slice_metrics(labels, probabilities, offsets, patient_ids, keep_ids, threshold):
    pieces_y = []
    pieces_p = []
    new_offsets = [0]
    for index, patient_id in enumerate(patient_ids):
        if str(patient_id) not in keep_ids:
            continue
        start, stop = int(offsets[index]), int(offsets[index + 1])
        pieces_y.append(np.asarray(labels[start:stop]))
        pieces_p.append(np.asarray(probabilities[start:stop]))
        new_offsets.append(new_offsets[-1] + (stop - start))
    if not pieces_y:
        raise ValueError("hospital slice is empty")
    return _operating(np.concatenate(pieces_y), np.concatenate(pieces_p), np.asarray(new_offsets), threshold)


def _operating(labels, probabilities, offsets, threshold):
    ranking = ranking_metrics(labels, probabilities)
    rates = binary_rates(labels, probabilities, threshold)
    return {
        "auroc": ranking["auroc"],
        "auprc": ranking["auprc"],
        "brier": ranking["brier"],
        "sensitivity": rates["sensitivity"],
        "specificity": rates["specificity"],
        "precision": rates["precision"],
        "f1": rates["f1"],
        "utility": float(utility_at_threshold(labels, probabilities, offsets, threshold)),
        "n_hours": ranking["n_hours"],
        "n_positive": ranking["n_positive_hours"],
        "n_negative": ranking["n_hours"] - ranking["n_positive_hours"],
        "threshold": float(threshold),
    }


def _observation_fractions(features, layout) -> dict[str, float]:
    fractions = {}
    n_rows = features.shape[0]
    for variable in INPUT_COLUMNS:
        column = layout.missing_column.get(variable)
        if column is None:
            fractions[variable] = 1.0
            continue
        missing = np.asarray(features[:, column])
        fractions[variable] = float(np.mean(missing < 0.5))
    if n_rows == 0:
        raise ValueError("split has no rows")
    return fractions


def _decide(checkpoint, phase4) -> dict:
    validation = {}
    for name in ("temporal_last", "flexible_xgboost"):
        block = checkpoint["validation"][name]
        full = block["full"]
        validation[name] = {
            "full_utility": full["utility"],
            "full_auprc": full["auprc"],
            "full_auroc": full["auroc"],
            "sparse_utility": sparse_utility(block["random_subsets"]),
        }
    decision = select_serving_model(validation)
    decision["inputs"] = validation
    return decision


def _write_manifest(root, checkpoint, spec, names, phase4) -> None:
    name = checkpoint["decision"]["model_name"]
    record = phase4["models"][name]
    preprocessor = load_json(root / record["preprocessor_path"])
    if preprocessor["feature_names"] != record["feature_names"]:
        raise ValueError("preprocessor feature order does not match the saved model")
    calibrator_path = root / record["calibration_path"]
    if not calibrator_path.is_file():
        raise FileNotFoundError(calibrator_path)
    model_path = root / record["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    version = f"phase5-{name}-{checkpoint['started_at']}"
    metadata = {
        "model_name": name,
        "model_version": version,
        "model_family": "xgboost",
        "role": "provisional_research_serving_candidate",
        "training_dataset_version": DATASET_VERSION,
        "feature_schema_version": feature_schema_version(names),
        "prediction_target": "SepsisLabel",
        "prediction_horizon": PREDICTION_HORIZON,
        "training_timestamp": checkpoint["started_at"],
        "source_model_path": record["model_path"],
        "expected_feature_order": record["feature_names"],
        "decision_threshold": float(record["validation_calibrated_operating_point"]["threshold"]),
        "best_iteration": int(record["hyperparameters"]["best_iteration"]),
        "flexible_input": name == "flexible_xgboost",
        "supported_input_variables": list(INPUT_COLUMNS),
        "calibration_kind": load_json(calibrator_path)["kind"],
        "external_validation_still_required": True,
    }
    manifest = {
        "contract_version": "1",
        "status": "trained",
        "deployment_status": "research",
        "role": "provisional_research_serving_candidate",
        "model_version": version,
        "model_family": "xgboost",
        "model_name": name,
        "prediction_horizon": PREDICTION_HORIZON,
        "feature_spec_path": FEATURE_SPEC_RELATIVE.as_posix(),
        "preprocessor_path": record["preprocessor_path"],
        "model_path": record["model_path"],
        "calibration_path": record["calibration_path"],
        "metadata_path": METADATA_RELATIVE.as_posix(),
        "feature_schema_version": metadata["feature_schema_version"],
        "decision_threshold": metadata["decision_threshold"],
        "flexible_input": metadata["flexible_input"],
        "supported_input_variables": list(INPUT_COLUMNS),
        "notes": (
            "Provisional research serving candidate selected in Phase 5 from validation "
            "utility and sparse-input utility. Phase 2 and Phase 4 artifacts were not overwritten."
        ),
    }
    _write(root / METADATA_RELATIVE, metadata)
    _write(root / MANIFEST_RELATIVE, manifest)
    checkpoint["manifest_path"] = MANIFEST_RELATIVE.as_posix()


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(checkpoint), indent=2), encoding="utf-8")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_phase5()
