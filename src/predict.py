"""模型推理与预测区间计算。

贝叶斯神经网络的核心用法是多次前向传播：每次 BayesianLinear 都会采样
一组权重，因此同一个输入会得到多个预测样本。样本均值作为点预测，
样本分布用于估计预测区间和模型不确定性。
"""

from __future__ import annotations

import numpy as np


def mc_predict(model, loader, device, mc_samples: int = 50) -> dict[str, np.ndarray]:
    """执行 Monte Carlo 前向传播。

    返回:
        samples: [mc_samples, 样本数, horizon]，每次权重采样得到的预测均值。
        target: 标准化空间中的真实目标。
        mean: MC 样本平均。
        std: 总标准差，包含模型不确定性和模型输出的 aleatoric 方差。
    """
    import torch

    # 这里故意使用 train() 而不是 eval()。BayesianLinear 在 forward 中采样，
    # 保持 train 模式也方便未来兼容 MC Dropout。
    model.train()
    all_means = []
    all_log_vars = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop("target")
            sample_means = []
            sample_vars = []
            for _ in range(mc_samples):
                mean, log_var = model(batch)
                sample_means.append(mean.detach().cpu().numpy())
                sample_vars.append(torch.exp(log_var).detach().cpu().numpy())
            all_means.append(np.stack(sample_means, axis=0))
            all_log_vars.append(np.stack(sample_vars, axis=0))
            targets.append(target.detach().cpu().numpy())

    mean_samples = np.concatenate(all_means, axis=1)
    aleatoric_vars = np.concatenate(all_log_vars, axis=1)
    targets_np = np.concatenate(targets, axis=0)
    mean = mean_samples.mean(axis=0)
    # 总不确定性 = 不同权重采样均值的方差 + 每次预测自身输出的噪声方差。
    total_var = mean_samples.var(axis=0) + aleatoric_vars.mean(axis=0)
    return {
        "samples": mean_samples,
        "target": targets_np,
        "mean": mean,
        "std": np.sqrt(np.maximum(total_var, 1e-8)),
    }


def interval_from_samples(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """从 MC 样本中计算给定置信水平的分位数区间。"""
    alpha = 1.0 - level
    return np.quantile(samples, alpha / 2, axis=0), np.quantile(samples, 1 - alpha / 2, axis=0)
