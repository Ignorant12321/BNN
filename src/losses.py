"""训练损失函数。

主模型输出两个张量：

- mean: 未来各预测步的功率均值。
- log_var: 未来各预测步的对数方差。

因此点预测误差之外，还可以用 Gaussian Negative Log Likelihood 约束
模型给出的概率分布。BayesianLinear 层额外提供 KL 散度，二者组合成
近似 ELBO 形式的训练目标。
"""

from __future__ import annotations

import math

import torch


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算高斯负对数似然。

    log_var 使用 clamp 限制范围，避免方差过小导致梯度爆炸，或方差过大
    导致数值溢出。返回值是 batch 和 horizon 上的平均损失。
    """
    log_var = torch.clamp(log_var, min=-10.0, max=6.0)
    inv_var = torch.exp(-log_var)
    return 0.5 * torch.mean(log_var + (target - mean) ** 2 * inv_var + math.log(2 * math.pi))


def elbo_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    kl: torch.Tensor,
    beta: float,
    num_batches: int,
) -> torch.Tensor:
    """ELBO 风格损失：概率预测误差 + 贝叶斯层 KL 正则。

    beta 控制 KL 项权重；num_batches 用于把整个 epoch 的 KL 正则平均到
    每个 mini-batch，避免 batch 数变化时 KL 项尺度改变太多。
    """
    return gaussian_nll(mean, log_var, target) + beta * kl / max(num_batches, 1)
