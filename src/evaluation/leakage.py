"""Automated checks that feature code cannot see future rows.

These checks use synthetic series. They do not certify a trained model, because
phase 1 does not train one. They do certify the feature functions that later
models are expected to call.
"""

from __future__ import annotations

import numpy as np

from src.features.temporal import (
    backward_mean,
    backward_slope,
    hour_delta,
    observation_state,
)


def run_leakage_checks() -> list[str]:
    passed: list[str] = []
    _future_edit_does_not_change_past_state()
    passed.append("observation_state_ignores_future_values")
    _future_edit_does_not_change_backward_mean()
    passed.append("backward_mean_ignores_future_values")
    _nan_is_not_zero_in_the_mean()
    passed.append("backward_mean_does_not_treat_nan_as_zero")
    _hour_delta_requires_two_real_measurements()
    passed.append("hour_delta_does_not_invent_a_value_for_a_missing_hour")
    _slope_window_is_backward()
    passed.append("backward_slope_ignores_future_values")
    _missing_history_is_not_a_zero_time_since()
    passed.append("time_since_measurement_is_nan_before_any_observation")
    return passed


def _assert_same_state(left, right) -> None:
    for name in (
        "raw",
        "missing",
        "last_value",
        "time_since_measurement",
        "previous_distinct_value",
        "change_from_previous_measurement",
        "change_direction",
        "measurements_in_window",
    ):
        a = getattr(left, name)
        b = getattr(right, name)
        if isinstance(a, float) and np.isnan(a):
            assert isinstance(b, float) and np.isnan(b)
        else:
            assert a == b


def _future_edit_does_not_change_past_state() -> None:
    series = np.array([2.4, np.nan, np.nan, np.nan, 9.0], dtype=float)
    before = observation_state(series, t=3)
    edited = series.copy()
    edited[4] = 100.0
    after = observation_state(edited, t=3)
    assert before.last_value == 2.4
    assert before.missing == 1
    assert before.time_since_measurement == 3
    assert np.isnan(before.raw)
    _assert_same_state(before, after)


def _future_edit_does_not_change_backward_mean() -> None:
    series = np.array([80, 82, 84, 86, 200], dtype=float)
    before = backward_mean(series, t=3, window=6)
    edited = series.copy()
    edited[4] = -50
    after = backward_mean(edited, t=3, window=6)
    assert before == after == 83.0


def _nan_is_not_zero_in_the_mean() -> None:
    series = np.array([80.0, np.nan, np.nan, 110.0, np.nan], dtype=float)
    # A zero-filled 5-hour mean would be 38. The observed-only mean is 95.
    assert backward_mean(series, t=4, window=6) == 95.0


def _hour_delta_requires_two_real_measurements() -> None:
    series = np.array([80.0, np.nan, np.nan, 110.0], dtype=float)
    assert np.isnan(hour_delta(series, t=3, lag=1))
    assert np.isnan(hour_delta(series, t=2, lag=1))
    assert hour_delta(np.array([80.0, 90.0]), 1, 1) == 10.0


def _slope_window_is_backward() -> None:
    series = np.array([0.0, 1.0, 2.0, 3.0, 100.0], dtype=float)
    slope = backward_slope(series, t=3, window=4)
    edited = series.copy()
    edited[4] = -100.0
    assert slope == backward_slope(edited, t=3, window=4) == 1.0


def _missing_history_is_not_a_zero_time_since() -> None:
    series = np.array([np.nan, np.nan, 4.0], dtype=float)
    early = observation_state(series, t=1)
    assert early.missing == 1
    assert np.isnan(early.last_value)
    assert np.isnan(early.time_since_measurement)
    current = observation_state(series, t=2)
    assert current.missing == 0
    assert current.time_since_measurement == 0
    assert current.last_value == 4.0
