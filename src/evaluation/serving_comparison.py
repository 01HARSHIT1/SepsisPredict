"""Validation rule for choosing the application model.

The score is fixed before the sparse-input results are read. Full-input
utility is the project objective. The sparse term is the mean utility of
deterministic random subsets of size 5 and 10, because the application must
accept arbitrary parameters. Test rows are not an input to this function.
"""

from __future__ import annotations

import numpy as np

from src.data.schema import LABS, VITALS
from src.inference.features import INPUT_COLUMNS
from src.training.flexible_mask import RAPID_VITALS, SPARSE_LABS
from src.training.masks import SELECTED_LABS

SUBSET_SEED = 2019
SUBSET_SIZES = (1, 2, 3, 5, 10, 20, 30, 40)
SUBSETS_PER_SIZE = 6
SPARSE_SCORE_SIZES = (5, 10)
ONE_VARIABLES = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "Lactate",
    "Creatinine",
    "WBC",
    "Platelets",
    "Bilirubin_total",
    "Glucose",
    "pH",
    "BUN",
    "Hgb",
    "Hct",
)
CANDIDATES = ("temporal_last", "flexible_xgboost")


def random_subsets(sizes: tuple[int, ...] = SUBSET_SIZES, seed: int = SUBSET_SEED) -> dict[int, list[list[str]]]:
    """Draw variable subsets without using labels or split membership.

    One generator, seed 2019, advances in increasing subset size. Size 40 has
    the single complete set. Smaller sizes draw six distinct subsets.
    """
    generator = np.random.default_rng(seed)
    variables = np.asarray(INPUT_COLUMNS)
    drawn: dict[int, list[list[str]]] = {}
    for size in sizes:
        if size == len(INPUT_COLUMNS):
            drawn[size] = [list(INPUT_COLUMNS)]
            continue
        if size > len(INPUT_COLUMNS):
            raise ValueError(f"subset size {size} exceeds the supported inputs")
        chosen: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        while len(chosen) < SUBSETS_PER_SIZE:
            pick = tuple(sorted(generator.choice(variables, size=size, replace=False).tolist()))
            if pick in seen:
                continue
            seen.add(pick)
            chosen.append(list(pick))
        drawn[size] = chosen
    return drawn


def user_scenarios(availability_ranked: list[str]) -> dict[str, list[str]]:
    """Clinical request shapes. The ranked list must come from training counts."""
    common_labs = [name for name in ("Lactate", "Creatinine", "WBC", "Platelets", "BUN", "Glucose") if name in SELECTED_LABS or name in LABS]
    return {
        "A_hr_only": ["HR"],
        "B_hr_resp": ["HR", "Resp"],
        "C_hr_resp_temp": ["HR", "Resp", "Temp"],
        "D_common_vitals": ["HR", "Resp", "Temp", "MAP", "SBP", "O2Sat"],
        "E_vitals_and_selected_labs": list(RAPID_VITALS) + common_labs,
        "F_vitals_and_sparse_labs": list(RAPID_VITALS) + list(SPARSE_LABS),
        "G_most_available": list(availability_ranked[:30]),
    }


def summarize_subset_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("subset summary needs at least one row")
    summary = {"n_subsets": len(rows)}
    for key in ("auroc", "auprc", "utility"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    highest = max(rows, key=lambda row: (row["utility"], row["auroc"]))
    lowest = min(rows, key=lambda row: (row["utility"], row["auroc"]))
    summary["highest_utility_subset"] = {
        "variables": highest["variables"],
        "utility": highest["utility"],
        "auroc": highest["auroc"],
        "auprc": highest["auprc"],
    }
    summary["lowest_utility_subset"] = {
        "variables": lowest["variables"],
        "utility": lowest["utility"],
        "auroc": lowest["auroc"],
        "auprc": lowest["auprc"],
    }
    return summary


def application_score(full_utility: float, sparse_utility: float) -> float:
    return 0.5 * float(full_utility) + 0.5 * float(sparse_utility)


def sparse_utility(random_subsets_by_size: dict) -> float:
    values = []
    for size in SPARSE_SCORE_SIZES:
        block = random_subsets_by_size[str(size)] if str(size) in random_subsets_by_size else random_subsets_by_size[size]
        values.extend(row["utility"] for row in block["subsets"])
    if not values:
        raise ValueError("sparse score needs subset utilities")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def select_serving_model(validation: dict) -> dict:
    """Choose from validation records only.

    ``validation[name]`` needs ``full_utility``, ``full_auprc``, ``full_auroc``,
    and ``sparse_utility``.
    """
    scored = []
    for name in CANDIDATES:
        record = validation[name]
        score = application_score(record["full_utility"], record["sparse_utility"])
        scored.append(
            {
                "model_name": name,
                "full_utility": float(record["full_utility"]),
                "full_auprc": float(record["full_auprc"]),
                "full_auroc": float(record["full_auroc"]),
                "sparse_utility": float(record["sparse_utility"]),
                "application_score": score,
            }
        )
    scored.sort(key=lambda row: (row["application_score"], row["full_auprc"], row["full_auroc"]), reverse=True)
    winner = scored[0]
    return {
        "model_name": winner["model_name"],
        "selection_split": "validation",
        "rule": (
            "application_score = 0.5 * pooled validation utility on natural inputs "
            "+ 0.5 * mean validation utility of random subsets of size 5 and 10. "
            "Ties use validation AUPRC, then validation AUROC. Test rows are not used."
        ),
        "candidates": scored,
    }


def dynamic_names() -> tuple[str, ...]:
    return tuple(VITALS) + tuple(LABS)
