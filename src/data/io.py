"""Readers for one-patient PhysioNet 2019 pipe-delimited files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.schema import EXPECTED_COLUMNS


def read_psv(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (header, float array) with missing entries as NaN.

    NaN tokens are preserved as missing. They are never replaced with zero.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"{path.name}: empty file")
        header = header_line.rstrip("\r\n").split("|")
        width = len(header)
        rows: list[np.ndarray] = []
        for line_number, line in enumerate(handle, start=2):
            line = line.rstrip("\r\n")
            if line == "":
                continue
            parts = line.split("|")
            if len(parts) != width:
                raise ValueError(f"{path.name}: line {line_number} has {len(parts)} fields, expected {width}")
            row = np.empty(width, dtype=np.float64)
            for index, token in enumerate(parts):
                if token == "" or token == "NaN" or token == "nan":
                    row[index] = np.nan
                else:
                    row[index] = float(token)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no data rows")
    return header, np.vstack(rows)


def validate_header(header: list[str], path: Path) -> None:
    if tuple(header) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path.name}: header does not match the official 41-column schema. Got {header}"
        )
