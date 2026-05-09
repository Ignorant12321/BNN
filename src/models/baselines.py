from __future__ import annotations

import torch
from torch import nn

from src.models.branches import HistoryCNNBranch, SequenceMLPBranch


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNBaseline(nn.Module):
    def __init__(self, history_features: int, horizon: int, hidden_dim: int = 64):
        super().__init__()
        self.history = HistoryCNNBranch(history_features, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.history(batch["history"]))


class CNNMLPBaseline(nn.Module):
    def __init__(self, history_features: int, weather_features: int, time_features: int, horizon: int, hidden_dim: int = 64):
        super().__init__()
        self.history = HistoryCNNBranch(history_features, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.weather = SequenceMLPBranch(weather_features, horizon, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.time = SequenceMLPBranch(time_features, horizon, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, horizon))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z = torch.cat([self.history(batch["history"]), self.weather(batch["weather"]), self.time(batch["time"])], dim=-1)
        return self.head(z)


class MCDropoutPVNet(CNNMLPBaseline):
    def __init__(self, history_features: int, weather_features: int, time_features: int, horizon: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__(history_features, weather_features, time_features, horizon, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
