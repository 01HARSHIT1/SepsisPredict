import numpy as np
import pytest

from src.evaluation.serving_comparison import (
    random_subsets,
    select_serving_model,
    summarize_subset_metrics,
)
from src.inference.features import parse_request
from src.inference.scoring import clinical_explanations
from src.inference.sufficiency import band_for_dynamic_count, information_descriptor


def test_random_subsets_are_deterministic_and_do_not_use_labels():
    first = random_subsets()
    second = random_subsets()
    assert first == second
    assert first[40] == [list(__import__("src.inference.features", fromlist=["INPUT_COLUMNS"]).INPUT_COLUMNS)]
    assert len(first[5]) == 6
    assert all(len(group) == 5 for group in first[5])
    assert len({tuple(group) for group in first[5]}) == 6


def test_selection_uses_validation_scores_only():
    chosen = select_serving_model(
        {
            "temporal_last": {
                "full_utility": 0.43,
                "full_auprc": 0.12,
                "full_auroc": 0.84,
                "sparse_utility": 0.05,
            },
            "flexible_xgboost": {
                "full_utility": 0.42,
                "full_auprc": 0.11,
                "full_auroc": 0.83,
                "sparse_utility": 0.20,
            },
        }
    )
    assert chosen["model_name"] == "flexible_xgboost"
    assert chosen["selection_split"] == "validation"
    assert "test" not in chosen["rule"]


def test_information_band_counts_dynamic_values_only():
    assert band_for_dynamic_count(0) == "VERY_LOW"
    assert band_for_dynamic_count(2) == "LOW"
    assert band_for_dynamic_count(3) == "MODERATE"
    assert band_for_dynamic_count(6) == "HIGH"
    measurements, _, _ = parse_request({"Age": 70, "Gender": 1, "ICULOS": 4})
    descriptor = information_descriptor(measurements, 0)
    assert descriptor["data_available"] == 3
    assert descriptor["data_sufficiency"] == "VERY_LOW"
    assert descriptor["observed_dynamic_variables"] == 0
    assert descriptor["has_prior_measurements"] is False


def test_non_chronological_hours_are_rejected():
    with pytest.raises(ValueError, match="chronological"):
        parse_request([{"hour": 3, "HR": 90}, {"hour": 1, "HR": 80}])
    with pytest.raises(ValueError, match="unique"):
        parse_request([{"hour": 1, "HR": 90}, {"hour": 1, "HR": 80}])


def test_explanations_group_engineered_columns_by_variable():
    rows = clinical_explanations(np.array([0.2, -0.5, 0.1]), ["HR_raw", "HR_missing", "MAP_raw"])
    assert rows[0]["feature"] == "HR"
    assert rows[0]["contribution"] == pytest.approx(-0.3)
    assert rows[0]["direction"] == "decreases_risk_score"
    assert rows[1]["feature"] == "MAP"
    assert "HR_raw" not in {row["feature"] for row in rows}


def test_subset_summary_reports_spread():
    summary = summarize_subset_metrics(
        [
            {"variables": ["HR"], "auroc": 0.6, "auprc": 0.03, "utility": 0.1},
            {"variables": ["Temp"], "auroc": 0.8, "auprc": 0.05, "utility": 0.3},
        ]
    )
    assert summary["mean_utility"] == pytest.approx(0.2)
    assert summary["std_utility"] == pytest.approx(0.1414213562373095)
    assert summary["highest_utility_subset"]["variables"] == ["Temp"]
    assert summary["lowest_utility_subset"]["variables"] == ["HR"]
