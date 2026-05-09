"""贝叶斯神经网络基础层。

普通 Linear 层的权重是确定值；BayesianLinear 中每个权重和偏置都被看作
一个高斯分布：

    w ~ Normal(mu, sigma)

训练时学习 mu 和 rho，其中 sigma = softplus(rho) 保证标准差为正。
每次 forward 都通过重参数化采样一组权重，从而让同一个模型可以产生
多个可能预测结果，用于刻画模型不确定性。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    """带高斯变分后验的全连接层。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ):
        """初始化贝叶斯线性层。

        prior_mu/prior_sigma 表示权重先验分布。当前实现使用标准正态附近
        的先验，KL loss 会约束后验不要离先验过远，从而起到正则化作用。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化可学习参数。

        mu 按普通 Linear 类似方式初始化；rho 初始化为 -5，使 softplus(rho)
        得到较小的标准差，训练初期采样扰动不会过大。
        """
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """采样权重和偏置后执行线性变换。"""
        weight = self._sample(self.weight_mu, self.weight_rho)
        bias = self._sample(self.bias_mu, self.bias_rho)
        return F.linear(x, weight, bias)

    def kl_loss(self) -> torch.Tensor:
        """返回该层权重和偏置后验相对先验的 KL 散度。"""
        return self._kl_normal(self.weight_mu, self.weight_rho) + self._kl_normal(self.bias_mu, self.bias_rho)

    @staticmethod
    def _sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """重参数化采样：sample = mu + sigma * eps。"""
        sigma = F.softplus(rho)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def _kl_normal(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """计算两个对角高斯分布之间的 KL 散度。"""
        sigma = F.softplus(rho)
        prior_sigma = torch.as_tensor(self.prior_sigma, dtype=mu.dtype, device=mu.device)
        prior_mu = torch.as_tensor(self.prior_mu, dtype=mu.dtype, device=mu.device)
        return torch.sum(
            torch.log(prior_sigma / sigma)
            + (sigma.pow(2) + (mu - prior_mu).pow(2)) / (2 * prior_sigma.pow(2))
            - 0.5
        )
