"""Phase 1 figures. Axes name the measured quantity and the cohort."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_figures(
    variable_table: pd.DataFrame,
    patients: pd.DataFrame,
    trajectory: dict,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _missingness(variable_table, output_dir / "missingness_by_variable.png"),
        _gaps(variable_table, output_dir / "median_hours_between_measurements.png"),
        _length_of_stay(patients, output_dir / "icu_hours_per_patient.png"),
        _trajectory(trajectory, output_dir / "mean_hr_map_before_first_positive_label.png"),
    ]
    return [path.name for path in written]


def _missingness(table: pd.DataFrame, path: Path) -> Path:
    clinical = table[table["role"].isin(["vital", "lab"])].copy()
    order = (
        clinical[clinical["hospital_system"] == "pooled"]
        .sort_values("pct_hours_missing", ascending=True)["variable"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 12))
    y = range(len(order))
    for hospital, color, offset in (("A", "#1f4e79", -0.15), ("B", "#b85c38", 0.15)):
        part = clinical[clinical["hospital_system"] == hospital].set_index("variable").loc[order]
        ax.barh([i + offset for i in y], part["pct_hours_missing"], height=0.3, color=color, label=f"Training set {hospital}")
    ax.set_yticks(list(y))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Hours with no recorded value (%)")
    ax.set_ylabel("Clinical variable")
    ax.set_xlim(0, 100)
    ax.set_title("Missing hours by variable, PhysioNet/CinC 2019 training sets A and B")
    ax.legend(loc="lower right")
    ax.figure.text(0.01, 0.01, "Source: training_setA and training_setB. NaN counted as not recorded. Not zero-filled.", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _gaps(table: pd.DataFrame, path: Path) -> Path:
    pooled = table[(table["hospital_system"] == "pooled") & (table["role"].isin(["vital", "lab"]))].copy()
    pooled = pooled.sort_values("median_gap_hours", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(pooled["variable"], pooled["median_gap_hours"].fillna(0), color="#1f4e79")
    ax.set_xlabel("Median hours between consecutive recorded values")
    ax.set_ylabel("Clinical variable")
    ax.set_title("Measurement spacing, pooled PhysioNet/CinC 2019 training data")
    ax.figure.text(
        0.01,
        0.01,
        "Gap = row-index difference between consecutive non-missing values. A gap of 1 hour means consecutive hourly records.",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _length_of_stay(patients: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = range(0, int(patients["n_hours"].max()) + 6, 6)
    for hospital, color in (("A", "#1f4e79"), ("B", "#b85c38")):
        part = patients.loc[patients["hospital_system"] == hospital, "n_hours"]
        ax.hist(part, bins=bins, alpha=0.55, label=f"Training set {hospital}", color=color)
    ax.set_xlabel("Record length (hours)")
    ax.set_ylabel("Patients")
    ax.set_title("ICU record length by training set")
    ax.legend()
    ax.figure.text(0.01, 0.01, "Each patient file contributes one length. Rows are hourly.", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _trajectory(trajectory: dict, path: Path) -> Path:
    offsets = trajectory["offsets_hours_relative_to_first_positive_label"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, variable, title in (
        (axes[0], "HR", "Mean heart rate"),
        (axes[1], "MAP", "Mean arterial pressure"),
    ):
        for hospital, color in (("A", "#1f4e79"), ("B", "#b85c38")):
            means = trajectory["hospitals"][hospital][variable]["mean"]
            ax.plot(offsets, [np_or_nan(value) for value in means], color=color, label=f"Training set {hospital}")
        ax.set_title(title)
        ax.set_xlabel("Hours relative to first SepsisLabel = 1")
        ax.set_ylabel("Mean of recorded values")
        ax.legend()
    fig.suptitle("Recorded values before the challenge early-label hour")
    fig.text(
        0.01,
        0.01,
        "Alignment is the first positive SepsisLabel, defined as t_sepsis - 6 hours, not onset. "
        "Patients positive from hour 1 are excluded. Means use recorded values only.",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def np_or_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value)
