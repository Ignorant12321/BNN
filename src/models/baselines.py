"""对比模型结构。

这些 baseline 用于论文中的消融或对比实验。为了复用主训练、评估和可视化
流水线，接入训练流程的 baseline 统一输出 ``(mean, log_var)``，并提供
``kl_loss()`` 方法。确定性 baseline 的不确定性来自可学习的逐 horizon
方差参数；MC Dropout baseline 在推理时还会通过 dropout 多次前向传播产生
模型不确定性。
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.branches import ForecastWeatherMLPBranch, HistoryCNNBranch


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
    """CNN 历史分支 + MLP 天气分支的确定性基线。"""

    def __init__(self, history_features: int, weather_features: int, horizon: int, hidden_dim: int = 64):
        super().__init__()
        self.history = HistoryCNNBranch(history_features, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.weather = ForecastWeatherMLPBranch(weather_features, horizon, out_dim=hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, horizon))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """拼接历史、天气和时间分支表示后输出点预测。"""
        z = torch.cat([self.history(batch["history"]), self.weather(batch["weather"])], dim=-1)
        return self.head(z)


class MCDropoutPVNet(CNNMLPBaseline):
    """MC Dropout 概率预测基线。

    推理时保持 dropout 激活，多次前向传播即可得到预测样本分布。它可以与
    BayesianLinear 的不确定性建模效果做对比。
    """

    def __init__(self, history_features: int, weather_features: int, horizon: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__(history_features, weather_features, horizon, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )


class ProbabilisticBaselineMixin:
    """为确定性 baseline 补齐主训练流程需要的概率预测接口。"""

    horizon: int
    log_var: nn.Parameter

    def _expand_log_var(self, mean: torch.Tensor) -> torch.Tensor:
        """把逐 horizon 方差参数扩展到当前 batch。"""
        return self.log_var.unsqueeze(0).expand_as(mean)

    def kl_loss(self) -> torch.Tensor:
        """确定性 baseline 没有 BayesianLinear，KL 项为 0。"""
        return self.log_var.new_zeros(())


class MLPProbabilisticBaseline(nn.Module, ProbabilisticBaselineMixin):
    """展平 history/weather/direct 后输入 MLP 的概率 baseline。"""

    def __init__(
        self,
        history_features: int,
        weather_features: int,
        direct_features: int,
        lookback: int,
        horizon: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        input_dim = history_features * lookback + weather_features * horizon + direct_features
        self.point_model = MLPBaseline(input_dim=input_dim, horizon=horizon, hidden_dim=hidden_dim)
        self.log_var = nn.Parameter(torch.zeros(horizon))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """返回未来 horizon 步的均值和可学习对数方差。"""
        parts = [
            batch["history"].flatten(start_dim=1),
            batch["weather"].flatten(start_dim=1),
        ]
        direct = batch["direct"]
        if direct.ndim == 1:
            direct = direct.unsqueeze(-1)
        parts.append(direct)
        mean = self.point_model(torch.cat(parts, dim=-1))
        return mean, self._expand_log_var(mean)


class CNNProbabilisticBaseline(CNNBaseline, ProbabilisticBaselineMixin):
    """只使用历史功率 CNN 的概率 baseline。"""

    def __init__(self, history_features: int, horizon: int, hidden_dim: int = 64):
        super().__init__(history_features=history_features, horizon=horizon, hidden_dim=hidden_dim)
        self.horizon = horizon
        self.log_var = nn.Parameter(torch.zeros(horizon))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 CNN 点预测均值和可学习对数方差。"""
        mean = super().forward(batch)
        return mean, self._expand_log_var(mean)


class CNNMLPProbabilisticBaseline(CNNMLPBaseline, ProbabilisticBaselineMixin):
    """CNN 历史分支 + MLP 天气分支的概率 baseline。"""

    def __init__(self, history_features: int, weather_features: int, horizon: int, hidden_dim: int = 64):
        super().__init__(history_features=history_features, weather_features=weather_features, horizon=horizon, hidden_dim=hidden_dim)
        self.horizon = horizon
        self.log_var = nn.Parameter(torch.zeros(horizon))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """返回多分支点预测均值和可学习对数方差。"""
        mean = super().forward(batch)
        return mean, self._expand_log_var(mean)


class MCDropoutProbabilisticBaseline(MCDropoutPVNet, ProbabilisticBaselineMixin):
    """MC Dropout 概率 baseline，接口与 BNN 主模型一致。"""

    def __init__(
        self,
        history_features: int,
        weather_features: int,
        horizon: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__(
            history_features=history_features,
            weather_features=weather_features,
            horizon=horizon,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.horizon = horizon
        self.log_var = nn.Parameter(torch.zeros(horizon))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 MC Dropout 均值样本和可学习对数方差。"""
        mean = super().forward(batch)
        return mean, self._expand_log_var(mean)
