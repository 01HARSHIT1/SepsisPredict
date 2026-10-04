"""Run the phase 1 audit, patient split, leakage checks, and report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.audit import audit_directory
from src.data.schema import SPLIT_SEED
from src.data.splits import assert_split_integrity, assign_patient_splits
from src.evaluation.leakage import run_leakage_checks
from src.inference.artifacts import write_phase1_artifacts
from src.evaluation.utility import assert_published_utility_example
from src.reporting.figures import save_figures
from src.reporting import dump_json, write_phase1_report


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _missingness_vs_label(pooled: pd.DataFrame) -> pd.DataFrame:
    work = pooled[pooled["role"].isin(["vital", "lab", "static"])].copy()
    rows = []
    for _, row in work.iterrows():
        miss = row["n_missing_label0"] + row["n_missing_label1"]
        obs = row["n_observed_label0"] + row["n_observed_label1"]
        p_miss = row["n_missing_label1"] / miss if miss else float("nan")
        p_obs = row["n_observed_label1"] / obs if obs else float("nan")
        rows.append(
            {
                "variable": row["variable"],
                "n_missing_hours": int(miss),
                "n_observed_hours": int(obs),
                "positive_label_rate_when_missing": p_miss,
                "positive_label_rate_when_observed": p_obs,
                "rate_difference_missing_minus_observed": p_miss - p_obs if miss and obs else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.reindex(frame["rate_difference_missing_minus_observed"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _attach_hospital_missingness(table: pd.DataFrame) -> pd.DataFrame:
    pooled = table[table["hospital_system"] == "pooled"].copy()
    for hospital in ("A", "B"):
        part = table[table["hospital_system"] == hospital][["variable", "pct_hours_missing", "pct_hours_observed", "median_gap_hours"]]
        part = part.rename(
            columns={
                "pct_hours_missing": f"pct_hours_missing_{hospital}",
                "pct_hours_observed": f"pct_hours_observed_{hospital}",
                "median_gap_hours": f"median_gap_hours_{hospital}",
            }
        )
        pooled = pooled.merge(part, on="variable", how="left")
    return pooled


def run() -> None:
    root = project_root()
    raw = root / "data" / "raw"
    result = audit_directory(raw)
    reports = root / "reports"
    tables = reports / "tables"
    figures = reports / "figures"
    processed = root / "data" / "processed"
    splits = root / "data" / "splits"
    for directory in (tables, figures, processed, splits):
        directory.mkdir(parents=True, exist_ok=True)

    result.variable_table.to_csv(tables / "variable_audit.csv", index=False)
    pooled = _attach_hospital_missingness(result.variable_table)
    pooled.to_csv(tables / "variable_audit_pooled.csv", index=False)
    pooled.to_csv(tables / "feature_dictionary.csv", index=False)
    write_phase1_artifacts(pooled, root=root)
    _missingness_vs_label(pooled).to_csv(tables / "missingness_vs_label.csv", index=False)
    result.correlations.to_csv(tables / "pairwise_pearson.csv", index=False)
    result.unit_counts.to_csv(tables / "unit_counts.csv", index=False)
    result.iculos_curves.to_csv(tables / "iculos_group_means.csv", index=False)
    _trajectory_frame(result.trajectory).to_csv(tables / "trajectory_means.csv", index=False)
    if result.read_errors:
        pd.DataFrame(result.read_errors).to_csv(tables / "read_errors.csv", index=False)
    result.patients.to_parquet(processed / "patient_summary.parquet", index=False)

    split = assign_patient_splits(result.patients, seed=SPLIT_SEED)
    assert_split_integrity(split.assignments)
    split.assignments.to_csv(splits / "patient_splits.csv", index=False)
    for name in ("train", "validation", "test"):
        ids = split.assignments.loc[split.assignments["split"] == name, "patient_id"]
        ids.to_csv(splits / f"{name}_patient_ids.csv", index=False, header=True)
    # Stratum counts make the 70/15/15 claim checkable.
    stratum = (
        split.assignments.groupby(["split", "hospital_system", "sepsis_patient"], as_index=False)
        .size()
        .rename(columns={"size": "n_patients"})
    )
    stratum.to_csv(splits / "split_stratum_counts.csv", index=False)
    (splits / "split_summary.json").write_text(json.dumps(split.summary, indent=2), encoding="utf-8")

    figure_names = save_figures(result.variable_table, result.patients, result.trajectory, figures)
    leakage = run_leakage_checks()
    utility_example = assert_published_utility_example()

    cohort_payload = {
        "cohort": result.cohort,
        "n_read_errors": len(result.read_errors),
        "split": split.summary,
        "leakage_checks_passed": leakage,
        "utility_published_example": utility_example,
        "figures": figure_names,
    }
    dump_json(reports / "dataset_audit_summary.json", cohort_payload)
    write_phase1_report(
        cohort=result.cohort,
        pooled=pooled,
        split_summary=split.summary,
        leakage_checks=leakage,
        utility_example=utility_example,
        read_errors=result.read_errors,
        unit_counts=result.unit_counts,
        output_path=reports / "phase1_report.md",
    )
    print(f"wrote {reports / 'phase1_report.md'}", flush=True)


def _trajectory_frame(trajectory: dict) -> pd.DataFrame:
    rows = []
    offsets = trajectory["offsets_hours_relative_to_first_positive_label"]
    for hospital, variables in trajectory["hospitals"].items():
        for variable, payload in variables.items():
            for offset, mean, n in zip(offsets, payload["mean"], payload["n"], strict=True):
                rows.append(
                    {
                        "hospital_system": hospital,
                        "variable": variable,
                        "hours_relative_to_first_positive_label": offset,
                        "n_recorded": n,
                        "mean": mean,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
