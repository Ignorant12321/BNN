"""PyTorch 光伏预测模型。

功能：
    提供使用 GPU/CUDA 训练的神经网络模型，接口统一返回 `(mean, log_var)`。

使用：
    python -m src.experiments.train --config configs/models/bnn_24h.yaml
"""

from __future__ import annotations

from typing import Literal

from src.torch_runtime import import_torch

torch = import_torch()
from torch import nn


FeatureMode = Literal["all", "history", "history_weather"]


class TorchPVNet(nn.Module):
    """通用 PyTorch 概率预测网络。"""

    is_torch_model = True

    def __init__(
        self,
        lookback: int,
        horizon: int,
        history_features: int,
        weather_features: int,
        direct_features: int,
        hidden_dim: int = 128,
        branch_dim: int = 64,
        feature_mode: FeatureMode = "all",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.feature_mode = feature_mode
        self.history_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * history_features, branch_dim),
            nn.ReLU(),
        )
        self.weather_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(horizon * weather_features, branch_dim),
            nn.ReLU(),
        )
        self.direct_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(direct_features, branch_dim),
            nn.ReLU(),
        )
        fusion_dim = self._fusion_dim(branch_dim, feature_mode)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, horizon)
        self.log_var_head = nn.Linear(hidden_dim, horizon)

    def forward(self, batch: dict) -> tuple:
        """前向传播，返回均值和对数方差。"""
        features = self._features(batch)
        z = self.fusion(features)
        mean = self.mean_head(z)
        log_var = torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)
        return mean, log_var

    def kl_loss(self):
        """当前普通 PyTorch 网络没有 BayesianLinear，KL 项为 0。"""
        return next(self.parameters()).new_zeros(())

    def _features(self, batch: dict):
        parts = []
        if self.feature_mode in {"all", "history", "history_weather"}:
            parts.append(self.history_branch(batch["history"]))
        if self.feature_mode in {"all", "history_weather"}:
            parts.append(self.weather_branch(batch["weather"]))
        if self.feature_mode == "all":
            parts.append(self.direct_branch(batch["direct"]))
        return torch.cat(parts, dim=1)

    @staticmethod
    def _fusion_dim(branch_dim: int, feature_mode: FeatureMode) -> int:
        dim = 0
        if feature_mode in {"all", "history", "history_weather"}:
            dim += branch_dim
        if feature_mode in {"all", "history_weather"}:
            dim += branch_dim
        if feature_mode == "all":
            dim += branch_dim
        return dim
