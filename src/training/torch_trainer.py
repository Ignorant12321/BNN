"""PyTorch/CUDA 训练器。

功能：
    将窗口数组转换为 TensorDataset，在指定设备上训练 PyTorch 模型，并计算指标。

使用：
    python -m src.experiments.train --config configs/models/bnn/24h.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.metrics import regression_metrics
from src.evaluation.predictor import predict_arrays
from src.data.scaling import inverse_target_prediction, inverse_target_values
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


def train_torch_model(
    model,
    arrays,
    config: dict[str, Any],
    epoch_callback=None,
    validation_arrays=None,
) -> list[dict[str, float]]:
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
    best_state_dict = None
    best_metric = float("inf")
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
            item = {"epoch": float(epoch_index + 1), "loss": float(np.mean(batch_losses))}
            if validation_arrays is not None:
                val_metrics = evaluate_torch_model(model, validation_arrays, device=device, config=config)
                model.train()
                item["val_rmse"] = val_metrics["rmse"]
                if val_metrics["rmse"] < best_metric:
                    best_metric = val_metrics["rmse"]
                    best_state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epoch_history.append(item)
            if epoch_callback is not None:
                epoch_callback(item)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
    return epoch_history


def evaluate_torch_model(model, arrays, device=None, config: dict[str, Any] | None = None) -> dict[str, float]:
    """在传入 split 的全部窗口上评估 PyTorch 模型。"""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    mean, log_var = predict_arrays(model, arrays, config=config)
    scaler = (config or {}).get("data", {}).get("scaling", {}).get("scaler")
    mean, log_var = inverse_target_prediction(mean, log_var, scaler)
    target = inverse_target_values(arrays.target, scaler)
    return regression_metrics(
        mean,
        log_var,
        target,
    )


def gaussian_nll(mean, log_var, target):
    """高斯负对数似然。"""
    inv_var = torch.exp(-log_var)
    return 0.5 * torch.mean(log_var + (target - mean) ** 2 * inv_var)
