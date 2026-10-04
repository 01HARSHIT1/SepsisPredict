"""Availability-ranked and structured feature masks.

Count masks keep the variables observed on the most training hours. Ties follow
the official column order. They are availability ranks, not ranks of predictive
importance. Structured masks are fixed lists. None of these choices is a claim
that the retained variables are clinically sufficient.
"""

from __future__ import annotations

from src.data.schema import LABS, STATIC, TIME_CONTEXT, VITALS
from src.inference.features import INPUT_COLUMNS

SELECTED_LABS = ("Lactate", "Creatinine", "WBC", "Platelets", "Bilirubin_total")


def rank_variables(raw_counts, n_hours: int) -> list[dict]:
    counts = [int(value) for value in raw_counts]
    if len(counts) != len(INPUT_COLUMNS):
        raise ValueError("observation counts must align with the 40 input columns")
    order = sorted(range(len(INPUT_COLUMNS)), key=lambda index: (-counts[index], index))
    ranked = []
    for rank, index in enumerate(order, start=1):
        ranked.append(
            {
                "rank": rank,
                "variable": INPUT_COLUMNS[index],
                "observed_hours": counts[index],
                "observation_rate": counts[index] / n_hours if n_hours else 0.0,
            }
        )
    return ranked


def mask_definitions(spec: dict, ranked: list[dict]) -> list[dict]:
    by_class: dict[str, list[str]] = {}
    for item in spec["features"]:
        by_class.setdefault(item["empirical_class"], [])
        if item["variable"] not in by_class[item["empirical_class"]]:
            by_class[item["empirical_class"]].append(item["variable"])
    variables = [item["variable"] for item in ranked]
    labs = set(LABS)
    conditions = [
        ("top_30", variables[:30]),
        ("top_20", variables[:20]),
        ("top_15", variables[:15]),
        ("top_10", variables[:10]),
        ("top_5", variables[:5]),
        ("top_3", variables[:3]),
        ("top_2", variables[:2]),
        ("top_1", variables[:1]),
        ("vitals_only", list(VITALS)),
        ("labs_only", list(LABS)),
        ("vitals_without_labs", [name for name in INPUT_COLUMNS if name not in labs]),
        ("selected_labs_without_vitals", list(SELECTED_LABS)),
        ("fast_vitals_only", by_class.get("near_hourly", [])),
        ("static_and_time", list(STATIC) + list(TIME_CONTEXT)),
        ("hr_only", ["HR"]),
        ("lactate_only", ["Lactate"]),
    ]
    masks = []
    for name, selected in conditions:
        if not selected:
            raise ValueError(f"{name} selected no variables")
        features = [item["name"] for item in spec["features"] if item["variable"] in set(selected)]
        masks.append(
            {
                "name": name,
                "variables": list(selected),
                "n_variables": len(selected),
                "feature_names": features,
                "n_features": len(features),
            }
        )
    return masks
