"""Patient-level alarm summaries for one fixed threshold.

Lead time is reconstructed onset index minus the first alarm index. Onset is
the challenge reconstruction, first positive label plus 6 hours. A positive
lead means the first alarm is before that onset. Very early alarms are
reported as a bin, not as a better clinical result.
"""

from __future__ import annotations

import numpy as np

from src.labels.targets import first_positive_index, reconstructed_onset_index


def lead_time_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    threshold: float,
) -> dict:
    predicted = np.asarray(probabilities) >= threshold
    leads: list[float] = []
    onset_inside = 0
    alarmed = 0
    bins = {
        "more_than_24h_before_onset": 0,
        "from_12h_to_24h_before_onset": 0,
        "from_6h_to_12h_before_onset": 0,
        "inside_official_six_hour_window": 0,
        "after_reconstructed_onset": 0,
        "never_alarmed": 0,
    }
    before_official_window = 0
    still_alarming_at_last_observed_hour = 0
    episode_counts: list[int] = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        y = np.asarray(labels[start:stop])
        onset = reconstructed_onset_index(y)
        if onset is None or onset < 0 or onset >= int(y.size):
            continue
        onset_inside += 1
        alarms = np.asarray(predicted[start:stop])
        episodes = _episodes(alarms)
        episode_counts.append(len(episodes))
        if not np.any(alarms):
            bins["never_alarmed"] += 1
            continue
        alarmed += 1
        first_alarm = int(np.flatnonzero(alarms)[0])
        lead = float(onset - first_alarm)
        leads.append(lead)
        first_positive = first_positive_index(y)
        if first_positive is not None and first_alarm < first_positive:
            before_official_window += 1
        if alarms[-1]:
            still_alarming_at_last_observed_hour += 1
        if lead > 24:
            bins["more_than_24h_before_onset"] += 1
        elif lead >= 12:
            bins["from_12h_to_24h_before_onset"] += 1
        elif lead > 6:
            bins["from_6h_to_12h_before_onset"] += 1
        elif lead >= 0:
            bins["inside_official_six_hour_window"] += 1
        else:
            bins["after_reconstructed_onset"] += 1
    array = np.asarray(leads, dtype=np.float64)
    edges = np.arange(-24, 97, 6)
    counts, edges = np.histogram(array, bins=edges)
    return {
        "threshold": float(threshold),
        "n_onset_inside_file": onset_inside,
        "n_with_alarm": alarmed,
        "lead_hours": _distribution(array),
        "lead_histogram": {
            "bin_edges_hours": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
        },
        "bins_among_onset_inside": bins,
        "bin_fractions_among_onset_inside": {
            key: (value / onset_inside if onset_inside else None) for key, value in bins.items()
        },
        "fraction_of_alarmed_before_official_positive_window": (
            before_official_window / alarmed if alarmed else None
        ),
        "fraction_of_alarmed_still_positive_at_record_end": (
            still_alarming_at_last_observed_hour / alarmed if alarmed else None
        ),
        "mean_alarm_episodes_among_onset_inside": (
            float(np.mean(episode_counts)) if episode_counts else None
        ),
        "definition": (
            "lead = reconstructed_onset_index - first hour at or above the threshold. "
            "Inside the official window means 0 <= lead <= 6. "
            "Edges: lead 6 is inside the window; lead 12 is in the 12-24 hour bin."
        ),
    }


def false_alarm_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    threshold: float,
) -> dict:
    """Alarm burden on patients who never have a positive SepsisLabel."""
    predicted = np.asarray(probabilities) >= threshold
    nonseptic = 0
    ever_alarmed = 0
    false_positive_hours = 0
    nonseptic_hours = 0
    episode_lengths: list[int] = []
    persistent_episodes = 0
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        y = np.asarray(labels[start:stop])
        if np.any(y == 1):
            continue
        nonseptic += 1
        alarms = np.asarray(predicted[start:stop])
        hours = int(alarms.size)
        nonseptic_hours += hours
        false_positive_hours += int(alarms.sum())
        if np.any(alarms):
            ever_alarmed += 1
        for length in _episodes(alarms):
            episode_lengths.append(length)
            if length >= 3:
                persistent_episodes += 1
    patient_days = nonseptic_hours / 24.0 if nonseptic_hours else None
    return {
        "threshold": float(threshold),
        "n_nonseptic_patients": nonseptic,
        "n_nonseptic_hours": nonseptic_hours,
        "false_positive_hours": false_positive_hours,
        "false_positive_hours_per_nonseptic_patient": (
            false_positive_hours / nonseptic if nonseptic else None
        ),
        "false_positive_hours_per_nonseptic_patient_day": (
            false_positive_hours / patient_days if patient_days else None
        ),
        "fraction_nonseptic_patients_ever_alarmed": (
            ever_alarmed / nonseptic if nonseptic else None
        ),
        "n_false_alarm_episodes": len(episode_lengths),
        "mean_false_alarm_episode_hours": (
            float(np.mean(episode_lengths)) if episode_lengths else None
        ),
        "persistent_episode_definition": "at least 3 consecutive hours at or above the threshold",
        "n_persistent_false_alarm_episodes": persistent_episodes,
        "fraction_false_alarm_episodes_persistent": (
            persistent_episodes / len(episode_lengths) if episode_lengths else None
        ),
    }


def _episodes(alarms: np.ndarray) -> list[int]:
    lengths: list[int] = []
    run = 0
    for value in alarms:
        if value:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def _distribution(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "p25": None,
            "p75": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }
