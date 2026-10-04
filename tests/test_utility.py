import numpy as np
import pytest

from src.evaluation.utility import assert_published_utility_example, compute_prediction_utility, normalized_utility


def test_published_example_matches_challenge_scorer():
    assert assert_published_utility_example() == pytest.approx(3.388888888888889)


def test_inaction_on_nonsepsis_is_zero():
    labels = np.zeros(8)
    assert compute_prediction_utility(labels, np.zeros(8)) == 0.0


def test_false_alarms_on_nonsepsis_are_penalized():
    labels = np.zeros(4)
    predictions = np.array([0, 1, 1, 0])
    assert compute_prediction_utility(labels, predictions) == pytest.approx(-0.10)


def test_normalized_utility_of_inaction_is_zero_and_best_is_one():
    labels = [np.array([0, 0, 0, 0, 1, 1, 1])]
    inaction = [np.zeros(7)]
    assert normalized_utility(labels, inaction) == pytest.approx(0.0)
    from src.evaluation.utility import best_predictions

    best = [best_predictions(labels[0])]
    assert normalized_utility(labels, best) == pytest.approx(1.0)
