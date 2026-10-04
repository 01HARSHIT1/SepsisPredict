import json

import numpy as np

from src.features.assemble import assemble_patient
from src.inference.features import INPUT_INDEX, build_feature_row


def test_assembled_matrix_matches_scalar_features_at_every_hour():
    spec = json.loads(open("artifacts/feature_definitions/feature_spec.json", encoding="utf-8").read())
    rng = np.random.default_rng(2019)
    measurements = rng.normal(size=(12, 40))
    mask = rng.random(measurements.shape) < 0.7
    measurements[mask] = np.nan
    measurements[:, INPUT_INDEX["Age"]] = 64.0
    measurements[0, INPUT_INDEX["Age"]] = np.nan
    assembled = assemble_patient(measurements, spec)
    names = [item["name"] for item in spec["features"]]
    for hour in (0, 1, 5, 11):
        scalar = build_feature_row(measurements, hour, spec)
        for index, name in enumerate(names):
            left = assembled[hour, index]
            right = scalar[name]
            if np.isnan(right):
                assert np.isnan(left), name
            else:
                assert left == np.float64(right), name
