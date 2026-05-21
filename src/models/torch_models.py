"""PyTorch 光伏预测模型。

功能：
    提供 improved_bnn、mlp_baseline、cnn_baseline、mc_dropout 四套独立
    PyTorch 模型，接口统一返回 `(mean, log_var)`。

使用：
    python -m src.experiments.train --config configs/models/bnn/24h.yaml
"""

from __future__ import annotations

import math

from src.torch_runtime import import_torch

torch = import_torch()
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    """均值场高斯后验的贝叶斯全连接层。"""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.prior_std = float(prior_std)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, inputs):
        weight = self._sample(self.weight_mu, self.weight_rho)
        bias = self._sample(self.bias_mu, self.bias_rho)
        return F.linear(inputs, weight, bias)

    def kl_loss(self):
        return gaussian_kl(self.weight_mu, self.weight_sigma, self.prior_std) + gaussian_kl(self.bias_mu, self.bias_sigma, self.prior_std)

    @property
    def weight_sigma(self):
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho)

    @staticmethod
    def _sample(mu, rho):
        sigma = F.softplus(rho)
        return mu + sigma * torch.randn_like(sigma)


class BayesianConv1d(nn.Module):
    """均值场高斯后验的一维贝叶斯卷积层。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        prior_std: float = 1.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.prior_std = float(prior_std)
        self.weight_mu = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.weight_rho = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias_mu = nn.Parameter(torch.empty(out_channels))
        self.bias_rho = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_channels * self.kernel_size)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, inputs):
        weight = self._sample(self.weight_mu, self.weight_rho)
        bias = self._sample(self.bias_mu, self.bias_rho)
        return F.conv1d(inputs, weight, bias=bias, padding=self.padding)

    def kl_loss(self):
        return gaussian_kl(self.weight_mu, self.weight_sigma, self.prior_std) + gaussian_kl(self.bias_mu, self.bias_sigma, self.prior_std)

    @property
    def weight_sigma(self):
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho)

    @staticmethod
    def _sample(mu, rho):
        sigma = F.softplus(rho)
        return mu + sigma * torch.randn_like(sigma)


def gaussian_kl(mu, sigma, prior_std: float = 1.0):
    """计算 q=N(mu,sigma) 到标准正态先验的 KL。"""
    prior_var = float(prior_std) ** 2
    return torch.sum(torch.log(torch.as_tensor(prior_std, device=mu.device, dtype=mu.dtype) / sigma) + (sigma**2 + mu**2) / (2 * prior_var) - 0.5)


class ProbabilisticTorchModel(nn.Module):
    """统一概率预测接口。"""

    is_torch_model = True
    stochastic_predict = False

    def kl_loss(self):
        total = next(self.parameters()).new_zeros(())
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "kl_loss"):
                total = total + module.kl_loss()
        return total


class ImprovedBayesianTorchNet(ProbabilisticTorchModel):
    """改进 BNN：按表 3 固定全连接分支、贝叶斯 1D 卷积分支和第三输入分支。"""

    stochastic_predict = True

    def __init__(
        self,
        lookback: int,
        horizon: int,
        history_features: int,
        weather_features: int,
        direct_features: int,
    ):
        super().__init__()
        self.horizon = int(horizon)
        history_flat_dim = int(lookback) * int(history_features)
        weather_flat_dim = int(horizon) * int(weather_features)
        self.history_fc = nn.Sequential(
            BayesianLinear(history_flat_dim, 32),
            nn.ReLU(),
            BayesianLinear(32, 64),
            nn.ReLU(),
            BayesianLinear(64, 16),
            nn.ReLU(),
        )
        self.history_conv1 = BayesianConv1d(history_features, 32, kernel_size=5, padding=2)
        self.history_conv2 = BayesianConv1d(32, 32, kernel_size=5, padding=2)
        self.conv_pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.conv_global_pool = nn.AdaptiveAvgPool1d(1)
        self.third_input_dim = weather_flat_dim + int(direct_features)
        fusion_dim = 16 + 32 + self.third_input_dim
        self.fusion = nn.Sequential(
            BayesianLinear(fusion_dim, 32),
            nn.ReLU(),
            BayesianLinear(32, 16),
            nn.ReLU(),
        )
        self.mean_head = BayesianLinear(16, horizon)
        self.log_var_head = BayesianLinear(16, horizon)

    def forward(self, batch: dict) -> tuple:
        history = batch["history"]
        history_flat = history.reshape(len(history), -1)
        fc_features = self.history_fc(history_flat)
        conv_inputs = history.transpose(1, 2)
        conv_features = F.relu(self.history_conv1(conv_inputs))
        conv_features = self.conv_pool(conv_features)
        conv_features = F.relu(self.history_conv2(conv_features))
        conv_features = self.conv_global_pool(conv_features).squeeze(-1)
        third_features = torch.cat(
            [batch["weather"].reshape(len(history), -1), batch["direct"].reshape(len(history), -1)],
            dim=1,
        )
        features = torch.cat([fc_features, conv_features, third_features], dim=1)
        z = self.fusion(features)
        return self.mean_head(z), torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)


class MLPBaselineTorchNet(ProbabilisticTorchModel):
    """普通 MLP baseline，使用全部输入但不包含贝叶斯层或卷积层。"""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        history_features: int,
        weather_features: int,
        direct_features: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.horizon = int(horizon)
        input_dim = int(lookback) * int(history_features) + int(horizon) * int(weather_features) + int(direct_features)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, horizon)
        self.log_var_head = nn.Linear(hidden_dim, horizon)

    def forward(self, batch: dict) -> tuple:
        features = torch.cat(
            [
                batch["history"].reshape(len(batch["history"]), -1),
                batch["weather"].reshape(len(batch["weather"]), -1),
                batch["direct"].reshape(len(batch["direct"]), -1),
            ],
            dim=1,
        )
        z = self.network(features)
        return self.mean_head(z), torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)


class CNNBaselineTorchNet(ProbabilisticTorchModel):
    """真正的一维卷积 baseline，只使用历史功率序列。"""

    def __init__(
        self,
        horizon: int,
        history_features: int,
        hidden_dim: int = 128,
        branch_dim: int = 64,
        conv_kernel: int = 5,
    ):
        super().__init__()
        self.horizon = int(horizon)
        padding = int(conv_kernel) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(history_features, branch_dim, kernel_size=conv_kernel, padding=padding),
            nn.ReLU(),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=conv_kernel, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, horizon)
        self.log_var_head = nn.Linear(hidden_dim, horizon)

    def forward(self, batch: dict) -> tuple:
        features = self.conv(batch["history"].transpose(1, 2))
        z = self.fusion(features)
        return self.mean_head(z), torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)


class MCDropoutTorchNet(ProbabilisticTorchModel):
    """MC Dropout baseline，预测阶段通过多次 dropout forward 估计不确定性。"""

    stochastic_predict = True

    def __init__(
        self,
        lookback: int,
        horizon: int,
        history_features: int,
        weather_features: int,
        hidden_dim: int = 128,
        branch_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.history_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(lookback) * int(history_features), branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.weather_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(horizon) * int(weather_features), branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, horizon)
        self.log_var_head = nn.Linear(hidden_dim, horizon)

    def forward(self, batch: dict) -> tuple:
        features = torch.cat([self.history_branch(batch["history"]), self.weather_branch(batch["weather"])], dim=1)
        z = self.fusion(features)
        return self.mean_head(z), torch.clamp(self.log_var_head(z), min=-10.0, max=6.0)
