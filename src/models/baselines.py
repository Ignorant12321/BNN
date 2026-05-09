"""对比模型结构。

这些 baseline 用于论文中的消融或对比实验。当前训练主流程优先支持
ImprovedBayesianPVNet；baseline 类已经放在这里，后续可以接入统一训练器。
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.branches import HistoryCNNBranch, SequenceMLPBranch


class MLPBaseline(nn.Module):
    """普通多层感知机基线。

    适合把所有输入特征展平后直接预测未来 horizon 步，作为最简单的深度学习
    对照组。
    """

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
        """返回 [batch, horizon] 点预测结果。"""
        return self.net(x)


class CNNBaseline(nn.Module):
    """只使用历史序列 CNN 的基线模型。"""

    def __init__(self, history_features: int, horizon: int, hidden_dim: int = 64):
        super().__init__()
        self.history = HistoryCNNBranch(history_features, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """从 batch 中读取 history 并预测未来序列。"""
        return self.head(self.history(batch["history"]))


class CNNMLPBaseline(nn.Module):
    """CNN 历史分支 + MLP 天气/时间分支的确定性基线。"""

    def __init__(self, history_features: int, weather_features: int, time_features: int, horizon: int, hidden_dim: int = 64):
        super().__init__()
        self.history = HistoryCNNBranch(history_features, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.weather = SequenceMLPBranch(weather_features, horizon, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.time = SequenceMLPBranch(time_features, horizon, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, horizon))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """拼接历史、天气和时间分支表示后输出点预测。"""
        z = torch.cat([self.history(batch["history"]), self.weather(batch["weather"]), self.time(batch["time"])], dim=-1)
        return self.head(z)


class MCDropoutPVNet(CNNMLPBaseline):
    """MC Dropout 概率预测基线。

    推理时保持 dropout 激活，多次前向传播即可得到预测样本分布。它可以与
    BayesianLinear 的不确定性建模效果做对比。
    """

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
