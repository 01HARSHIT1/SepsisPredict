"""Train one GRU window on the existing patient-level matrices.

Inputs are the causal feature columns. Non-finite measurements are scaled and
then filled with 0 while missing indicators remain separate.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from src.models.gru_inputs import missing_pad_row, to_network_inputs
from src.models.gru_net import HourlyGRU
from src.models.sequences import windows_ending_at_each_hour

HIDDEN_SIZE = 48
MAX_EPOCHS = 4


def fit_streaming_scaler(features: np.ndarray, offsets: np.ndarray, column_indexes: list[int], names: list[str]):
    count = np.zeros(len(names), dtype=np.float64)
    total = np.zeros(len(names), dtype=np.float64)
    squares = np.zeros(len(names), dtype=np.float64)
    for patient in range(len(offsets) - 1):
        block = np.array(features[int(offsets[patient]) : int(offsets[patient + 1])][:, column_indexes], dtype=np.float64)
        for index, name in enumerate(names):
            if name.endswith("_missing"):
                continue
            values = block[:, index]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            count[index] += values.size
            total[index] += values.sum()
            squares[index] += np.dot(values, values)
    scaler: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(names):
        if name.endswith("_missing") or count[index] == 0:
            scaler[name] = (0.0, 1.0)
            continue
        mean = total[index] / count[index]
        variance = squares[index] / count[index] - mean * mean
        scale = float(np.sqrt(variance)) if variance > 0 else 1.0
        scaler[name] = (float(mean), scale)
    return scaler


def train_gru(
    train_features,
    train_labels,
    train_offsets,
    valid_features,
    valid_labels,
    valid_offsets,
    column_indexes: list[int],
    names: list[str],
    window: int,
    seed: int = 2019,
) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    scaler = fit_streaming_scaler(train_features, train_offsets, column_indexes, names)
    pad = missing_pad_row(names, scaler)
    model = HourlyGRU(len(names), HIDDEN_SIZE)
    positive = float(np.asarray(train_labels).sum())
    negative = float(len(train_labels) - positive)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = copy.deepcopy(model.state_dict())
    best_auprc = -1.0
    stale = 0
    history = []
    patient_order = np.arange(len(train_offsets) - 1)
    generator = np.random.default_rng(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        generator.shuffle(patient_order)
        window_batch: list[np.ndarray] = []
        label_batch: list[np.ndarray] = []
        pending = 0
        for patient in patient_order:
            network, labels = _patient_windows(
                train_features, train_labels, train_offsets, int(patient), column_indexes, names, scaler, pad, window
            )
            window_batch.append(network)
            label_batch.append(labels)
            pending += labels.shape[0]
            if pending >= 512:
                _step(model, optimizer, loss_fn, window_batch, label_batch)
                window_batch, label_batch, pending = [], [], 0
        if pending:
            _step(model, optimizer, loss_fn, window_batch, label_batch)
        probabilities = predict_gru(
            model, valid_features, valid_offsets, column_indexes, names, scaler, pad, window
        )
        auprc = float(average_precision_score(np.asarray(valid_labels), probabilities))
        history.append({"epoch": epoch + 1, "validation_auprc": auprc})
        print(f"gru window {window} epoch {epoch + 1} validation AUPRC {auprc:.5f}", flush=True)
        if auprc > best_auprc:
            best_auprc = auprc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    model.load_state_dict(best_state)
    return {
        "model": model,
        "scaler": scaler,
        "pad": pad,
        "history": history,
        "best_validation_auprc": best_auprc,
        "hidden_size": HIDDEN_SIZE,
    }


def predict_gru(model, features, offsets, column_indexes, names, scaler, pad, window) -> np.ndarray:
    model.eval()
    output = np.empty(int(offsets[-1]), dtype=np.float64)
    with torch.no_grad():
        for patient in range(len(offsets) - 1):
            network, _labels = _patient_windows(
                features,
                None,
                offsets,
                patient,
                column_indexes,
                names,
                scaler,
                pad,
                window,
            )
            start = int(offsets[patient])
            stop = int(offsets[patient + 1])
            logits = model(torch.from_numpy(network)).numpy()
            output[start:stop] = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    return output


def _patient_windows(features, labels, offsets, patient, column_indexes, names, scaler, pad, window):
    start = int(offsets[patient])
    stop = int(offsets[patient + 1])
    block = np.array(features[start:stop][:, column_indexes], dtype=np.float32)
    network_rows = to_network_inputs(block, names, scaler)
    windows = np.ascontiguousarray(windows_ending_at_each_hour(network_rows, window, pad))
    if labels is None:
        return windows, None
    return windows, np.asarray(labels[start:stop], dtype=np.float32)


def _step(model, optimizer, loss_fn, window_batch, label_batch) -> None:
    inputs = torch.from_numpy(np.concatenate(window_batch))
    targets = torch.from_numpy(np.concatenate(label_batch))
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
