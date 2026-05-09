"""模型输入端的多分支特征提取模块。

论文中的改进点是：不同类型输入具有不同数据特性，不宜全部直接拼接后交给
同一个网络处理。本文件把输入分成三个可复用分支：

- HistoryCNNBranch: 处理历史出力等时间序列，提取局部波动和趋势。
- SequenceMLPBranch: 处理未来窗口的天气/时间序列特征。
- DirectInputBranch: 处理预测点前一时刻的强相关变量。
"""

from __future__ import annotations

import torch
from torch import nn


class HistoryCNNBranch(nn.Module):
    """历史序列 1D-CNN 分支。

    输入 shape 为 [batch, lookback, features]，Conv1d 要求通道在第二维，
    因此前向传播中会转成 [batch, features, lookback]。
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(hidden_dim, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取历史序列的定长表示。"""
        x = x.transpose(1, 2)
        return self.proj(self.net(x))


class SequenceMLPBranch(nn.Module):
    """预测窗口序列 MLP 分支。

    天气和时间特征已经按 horizon 对齐。这里直接 flatten 成一个向量，
    让 MLP 学习未来窗口内各步特征与目标序列之间的非线性关系。
    """

    def __init__(self, in_features: int, horizon: int, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features * horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把 [batch, horizon, features] 编码为 [batch, out_dim]。"""
        return self.net(x)


class DirectInputBranch(nn.Module):
    """最近时刻直接输入分支。"""

    def __init__(self, in_features: int, hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码预测点前一时刻的功率和辐照度等变量。"""
        return self.net(x)
