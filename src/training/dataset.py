"""Build the phase 2 matrices from the phase 1 patient splits.

Each row is one ICU hour. Features come from ``assemble_patient``, which is the
same function inference uses. Hours stay in file order inside each patient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import read_psv, validate_header
from src.features.assemble import assemble_patient
from src.inference.artifacts import FEATURE_SPEC_RELATIVE, load_json, project_root
from src.inference.features import DATA_TOTAL, INPUT_COLUMNS

PHASE2_RELATIVE = Path("data") / "processed" / "phase2"
RESERVOIR_SIZE = 200_000
RESERVOIR_SEED = 2019


def phase2_dir(root: Path | None = None) -> Path:
    root = project_root() if root is None else root
    return root / PHASE2_RELATIVE


def raw_patient_path(root: Path, patient_id: str) -> Path:
    number = int(patient_id[1:])
    folder = "training_setB" if number >= 100000 else "training_setA"
    return root / "data" / "raw" / folder / f"{patient_id}.psv"


def build_phase2_matrices(
    root: Path | None = None,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Write float32 feature memmaps for train, validation, and test.

    ``limit`` keeps the first patients of each split and is only for a smoke
    check. The training run uses every patient.
    """
    root = project_root() if root is None else root
    spec = load_json(root / FEATURE_SPEC_RELATIVE)
    feature_names = [item["name"] for item in spec["features"]]
    splits = pd.read_csv(root / "data" / "splits" / "patient_splits.csv")
    summary = pd.read_parquet(root / "data" / "processed" / "patient_summary.parquet")
    merged = splits.merge(summary[["patient_id", "n_hours"]], on="patient_id", how="inner")
    if len(merged) != len(splits):
        raise ValueError("patient splits and the patient summary do not contain the same ids")
    destination = phase2_dir(root) if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    written = {}
    for split_name in ("train", "validation", "test"):
        part = merged[merged["split"] == split_name].sort_values("patient_id")
        if limit is not None:
            part = part.head(limit)
        written[split_name] = _write_split(
            root,
            destination / split_name,
            part,
            spec,
            feature_names,
            collect_train_statistics=split_name == "train",
        )
    return written


def load_split(split_name: str, root: Path | None = None) -> dict:
    root = project_root() if root is None else root
    directory = phase2_dir(root) / split_name
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("complete"):
        raise ValueError(f"{split_name} matrix is incomplete")
    features = np.memmap(
        directory / "features.dat",
        dtype=np.float32,
        mode="r",
        shape=(meta["n_rows"], meta["n_features"]),
    )
    labels = np.memmap(directory / "labels.dat", dtype=np.uint8, mode="r", shape=(meta["n_rows"],))
    offsets = np.load(directory / "patient_offsets.npy")
    patient_ids = json.loads((directory / "patient_ids.json").read_text(encoding="utf-8"))
    return {
        "features": features,
        "labels": labels,
        "offsets": offsets,
        "patient_ids": patient_ids,
        "meta": meta,
    }


def load_train_statistics(root: Path | None = None) -> dict:
    root = project_root() if root is None else root
    directory = phase2_dir(root) / "train"
    return {
        "medians": np.load(directory / "feature_medians.npy"),
        "median_observations": np.load(directory / "feature_median_observations.npy"),
        "raw_observation_counts": np.load(directory / "raw_observation_counts.npy"),
        "raw_names": json.loads((directory / "raw_names.json").read_text(encoding="utf-8")),
    }


def _write_split(
    root: Path,
    directory: Path,
    patients: pd.DataFrame,
    spec: dict,
    feature_names: list[str],
    collect_train_statistics: bool,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "meta.json"
    n_rows = int(patients["n_hours"].sum())
    n_features = len(feature_names)
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            existing.get("complete")
            and existing.get("n_rows") == n_rows
            and existing.get("n_features") == n_features
            and existing.get("n_patients") == int(len(patients))
        ):
            return existing
    features = np.memmap(
        directory / "features.dat", dtype=np.float32, mode="w+", shape=(n_rows, n_features)
    )
    labels = np.memmap(directory / "labels.dat", dtype=np.uint8, mode="w+", shape=(n_rows,))
    offsets = np.zeros(len(patients) + 1, dtype=np.int64)
    patient_ids: list[str] = []
    raw_counts = np.zeros(DATA_TOTAL, dtype=np.int64)
    reservoir = _Reservoir(n_features, RESERVOIR_SIZE, RESERVOIR_SEED) if collect_train_statistics else None
    cursor = 0
    for position, row in enumerate(patients.itertuples(index=False), start=1):
        path = raw_patient_path(root, row.patient_id)
        header, values = read_psv(path)
        validate_header(header, path)
        if values.shape != (int(row.n_hours), DATA_TOTAL + 1):
            raise ValueError(f"{row.patient_id}: file length {values.shape[0]} does not match the audit")
        measurements = values[:, :DATA_TOTAL]
        target = values[:, DATA_TOTAL]
        if not np.isin(target, (0.0, 1.0)).all():
            raise ValueError(f"{row.patient_id}: SepsisLabel is not binary")
        block = assemble_patient(measurements, spec).astype(np.float32, copy=False)
        stop = cursor + block.shape[0]
        features[cursor:stop] = block
        labels[cursor:stop] = target.astype(np.uint8)
        raw_counts += np.isfinite(measurements).sum(axis=0).astype(np.int64)
        if reservoir is not None:
            reservoir.add(block)
        cursor = stop
        offsets[position] = cursor
        patient_ids.append(row.patient_id)
        if position % 2000 == 0 or position == len(patients):
            print(f"{directory.name}: {position}/{len(patients)} patients, {cursor} hours", flush=True)
    if cursor != n_rows:
        raise ValueError(f"{directory.name}: wrote {cursor} hours, expected {n_rows}")
    features.flush()
    labels.flush()
    del features, labels
    np.save(directory / "patient_offsets.npy", offsets)
    (directory / "patient_ids.json").write_text(json.dumps(patient_ids), encoding="utf-8")
    (directory / "feature_names.json").write_text(json.dumps(feature_names), encoding="utf-8")
    meta = {
        "complete": True,
        "split": directory.name,
        "n_rows": n_rows,
        "n_features": n_features,
        "n_patients": len(patient_ids),
        "n_positive_hours": _positive_hours(directory, n_rows),
        "n_sepsis_patients": int(patients["sepsis_patient"].sum()),
        "feature_dtype": "float32",
        "label": "SepsisLabel",
    }
    if reservoir is not None:
        medians, observed = reservoir.medians()
        np.save(directory / "feature_medians.npy", medians)
        np.save(directory / "feature_median_observations.npy", observed)
        np.save(directory / "raw_observation_counts.npy", raw_counts)
        (directory / "raw_names.json").write_text(json.dumps(list(INPUT_COLUMNS)), encoding="utf-8")
        meta["median_reservoir_size"] = RESERVOIR_SIZE
        meta["median_reservoir_seed"] = RESERVOIR_SEED
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _positive_hours(directory: Path, n_rows: int) -> int:
    labels = np.memmap(directory / "labels.dat", dtype=np.uint8, mode="r", shape=(n_rows,))
    return int(labels.sum())


class _Reservoir:
    def __init__(self, n_features: int, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.values = np.full((n_features, capacity), np.nan, dtype=np.float32)
        self.seen = np.zeros(n_features, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def add(self, block: np.ndarray) -> None:
        for column in range(block.shape[1]):
            observed = block[:, column]
            observed = observed[np.isfinite(observed)]
            if observed.size == 0:
                continue
            seen = int(self.seen[column])
            if seen < self.capacity:
                take = min(self.capacity - seen, int(observed.size))
                self.values[column, seen : seen + take] = observed[:take]
                observed = observed[take:]
                seen += take
            if observed.size:
                positions = seen + np.arange(1, observed.size + 1)
                keep = self.rng.random(observed.size) < (self.capacity / positions)
                chosen = observed[keep]
                if chosen.size:
                    slots = self.rng.integers(0, self.capacity, size=chosen.size)
                    self.values[column, slots] = chosen
                seen += int(observed.size)
            self.seen[column] = seen

    def medians(self) -> tuple[np.ndarray, np.ndarray]:
        medians = np.zeros(self.values.shape[0], dtype=np.float32)
        for column in range(self.values.shape[0]):
            count = min(int(self.seen[column]), self.capacity)
            if count == 0:
                medians[column] = 0.0
            else:
                medians[column] = np.median(self.values[column, :count])
        return medians, self.seen.copy()
