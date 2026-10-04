"""One-layer GRU over a recent hourly window."""

from __future__ import annotations

import torch
from torch import nn


class HourlyGRU(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _hidden = self.gru(inputs)
        return self.head(output[:, -1, :]).squeeze(-1)
