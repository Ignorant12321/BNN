"""改进贝叶斯光伏预测主模型。

模型结构对应论文中的“输入端分支改进 + 概率层融合”思想：

history -> 1D-CNN
weather -> MLP
direct  -> raw previous-step AC_POWER

三路输入的表示拼接后进入 BayesianLinear 层。模型最终输出两个张量：

- mean: 未来 horizon 步的预测均值。
- log_var: 未来 horizon 步的对数方差，用于描述数据噪声不确定性。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.bayesian_layers import BayesianLinear
from src.models.branches import ForecastWeatherMLPBranch, HistoryCNNBranch


class ImprovedBayesianPVNet(nn.Module):
    """用于 4h 超短期光伏概率预测的改进 BNN。"""

    def __init__(
        self,
        history_features: int,
        weather_features: int,
        direct_features: int,
        horizon: int,
        hidden_dim: int = 128,
        branch_dim: int = 64,
        prior_sigma: float = 1.0,
    ):
        """构建模型。

        参数中的 feature 数量来自 `src.features.split_feature_columns()`，
        horizon 默认为 16，对应 15 分钟粒度下的未来 4 小时。
        """
        super().__init__()
        self.horizon = horizon
        self.history_branch = HistoryCNNBranch(history_features, hidden_dim=branch_dim, out_dim=branch_dim)
        self.weather_branch = ForecastWeatherMLPBranch(weather_features, horizon=horizon, out_dim=branch_dim)
        # direct 分支按论文图作为第三部分输入，直接把 t-1 的 AC_POWER 拼入融合层。
        fusion_dim = branch_dim * 2 + direct_features
        self.bayes1 = BayesianLinear(fusion_dim, hidden_dim, prior_sigma=prior_sigma)
        self.bayes2 = BayesianLinear(hidden_dim, hidden_dim, prior_sigma=prior_sigma)
        self.mean_head = BayesianLinear(hidden_dim, horizon, prior_sigma=prior_sigma)
        self.log_var_head = BayesianLinear(hidden_dim, horizon, prior_sigma=prior_sigma)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """执行一次前向传播。

        batch 必须包含 history/weather/direct 三个键。返回值的 shape 均为
        [batch, horizon]。
        """
        direct = batch["direct"]
        if direct.ndim == 1:
            direct = direct.unsqueeze(-1)
        z = torch.cat(
            [
                self.history_branch(batch["history"]),
                self.weather_branch(batch["weather"]),
                direct,
            ],
            dim=-1,
        )
        z = F.relu(self.bayes1(z))
        z = F.relu(self.bayes2(z))
        mean = self.mean_head(z)
        # 限制 log_var 范围可以避免训练时出现极端方差导致的数值不稳定。
        log_var = torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)
        return mean, log_var

    def kl_loss(self) -> torch.Tensor:
        """汇总模型中所有 BayesianLinear 层的 KL 散度。"""
        kl = torch.zeros((), device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_loss()
        return kl
