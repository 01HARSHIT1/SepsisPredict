"""Score one request with a saved GRU window model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.features.assemble import assemble_patient
from src.inference.artifacts import load_json
from src.inference.features import parse_request
from src.inference.scoring import apply_calibrator
from src.models.gru_inputs import missing_pad_row, to_network_inputs
from src.models.gru_net import HourlyGRU
from src.models.sequences import windows_ending_at_each_hour


def score_gru_payload(payload: dict | list, manifest: dict, root: Path, spec: dict) -> np.ndarray:
    config = load_json(root / manifest["preprocessor_path"])
    measurements, scored, _hours = parse_request(payload)
    assembled = assemble_patient(measurements, spec)
    spec_names = [item["name"] for item in spec["features"]]
    indexes = [spec_names.index(name) for name in config["feature_names"]]
    scaler = {
        name: (float(mean), float(scale))
        for name, mean, scale in zip(
            config["feature_names"], config["scaler_mean"], config["scaler_scale"], strict=True
        )
    }
    network = to_network_inputs(assembled[:, indexes], config["feature_names"], scaler)
    pad = missing_pad_row(config["feature_names"], scaler)
    windows = windows_ending_at_each_hour(network, int(config["window"]), pad)
    model = HourlyGRU(len(config["feature_names"]), int(config["hidden_size"]))
    state = torch.load(root / config["state_dict_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    batch = torch.from_numpy(np.ascontiguousarray(windows[scored], dtype=np.float32))
    with torch.no_grad():
        logits = model(batch).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    calibrator = None
    if manifest.get("calibration_path"):
        calibrator = load_json(root / manifest["calibration_path"])
    return apply_calibrator(probabilities, calibrator)
