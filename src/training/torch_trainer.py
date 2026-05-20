"""PyTorch/CUDA 训练器。

功能：
    将窗口数组转换为 TensorDataset，在指定设备上训练 PyTorch 模型，并计算指标。

使用：
    python -m src.experiments.train --config configs/models/bnn_24h.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.torch_runtime import import_torch

torch = import_torch()
from torch.utils.data import DataLoader, TensorDataset


def resolve_torch_device(device_name: str = "auto"):
    """根据配置选择 torch 设备。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用 cuda，但当前 PyTorch 环境没有可用 CUDA")
    return torch.device(device_name)


def arrays_to_torch_dataset(arrays) -> TensorDataset:
    """把 WindowArrays 转为 TensorDataset。"""
    return TensorDataset(
        torch.from_numpy(arrays.history.astype(np.float32)),
        torch.from_numpy(arrays.weather.astype(np.float32)),
        torch.from_numpy(arrays.direct.astype(np.float32)),
        torch.from_numpy(arrays.target.astype(np.float32)),
    )


def train_torch_model(model, arrays, config: dict[str, Any]) -> list[dict[str, float]]:
    """在配置指定设备上训练 PyTorch 模型，并返回每轮训练损失。"""
    training = config.get("training", {})
    device = resolve_torch_device(str(training.get("device", "auto")))
    model.to(device)
    model.train()
    loader = DataLoader(
        arrays_to_torch_dataset(arrays),
        batch_size=int(training.get("batch_size", 64)),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    kl_beta = float(training.get("kl_beta", 0.0))
    epoch_history: list[dict[str, float]] = []
    for epoch_index in range(int(training.get("epochs", 20))):
        batch_losses = []
        for history_batch, weather, direct, target in loader:
            batch = {
                "history": history_batch.to(device),
                "weather": weather.to(device),
                "direct": direct.to(device),
            }
            target = target.to(device)
            mean, log_var = model(batch)
            loss = gaussian_nll(mean, log_var, target) + kl_beta * model.kl_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        if batch_losses:
            epoch_history.append({"epoch": float(epoch_index + 1), "loss": float(np.mean(batch_losses))})
    return epoch_history


def evaluate_torch_model(model, arrays, device=None) -> dict[str, float]:
    """在传入 split 的全部窗口上评估 PyTorch 模型。"""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch = {
            "history": torch.from_numpy(arrays.history.astype(np.float32)).to(device),
            "weather": torch.from_numpy(arrays.weather.astype(np.float32)).to(device),
            "direct": torch.from_numpy(arrays.direct.astype(np.float32)).to(device),
        }
        target = torch.from_numpy(arrays.target.astype(np.float32)).to(device)
        mean, log_var = model(batch)
        rmse = torch.sqrt(torch.mean((mean - target) ** 2))
        nll = gaussian_nll(mean, log_var, target)
    return {"rmse": float(rmse.detach().cpu()), "nll": float(nll.detach().cpu())}


def gaussian_nll(mean, log_var, target):
    """高斯负对数似然。"""
    inv_var = torch.exp(-log_var)
    return 0.5 * torch.mean(log_var + (target - mean) ** 2 * inv_var)
