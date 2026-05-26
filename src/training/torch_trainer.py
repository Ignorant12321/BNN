"""PyTorch/CUDA 训练器。

功能：
    将窗口数组转换为 TensorDataset，在指定设备上训练 PyTorch 模型，并计算指标。

使用：
    python -m src.experiments.train --config configs/models/bnn/24h.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.metrics import generation_period_metrics, prediction_frame_metrics, regression_metrics
from src.evaluation.predictor import predict_arrays, predict_dataframe
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
        foreach=bool(training.get("optimizer_foreach", False)),
    )
    kl_beta = float(training.get("kl_beta", 0.0))
    epoch_history: list[dict[str, float]] = []
    best_state_dict = None
    best_metric = float("inf")
    patience = early_stopping_patience(training)
    min_delta = early_stopping_min_delta(training)
    monitor_metric = early_stopping_monitor_metric(training)
    stale_epochs = 0
    for epoch_index in range(int(training.get("epochs", 20))):
        batch_losses = []
        for history_batch, weather, direct, target in loader:
            batch = {
                "history": history_batch.to(device),
                "weather": weather.to(device),
                "direct": direct.to(device),
                "target": target.to(device),
            }
            target = batch["target"]
            output = model(batch)
            if getattr(model, "deterministic_predict", False):
                mean = output
                loss = torch.mean((target - mean) ** 2)
            else:
                mean, log_var = output
                loss = gaussian_nll(mean, log_var, target) + kl_beta * model.kl_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        if batch_losses:
            item = {"epoch": float(epoch_index + 1), "loss": float(np.mean(batch_losses))}
            if validation_arrays is not None:
                val_metrics = evaluate_torch_validation_metrics(model, validation_arrays, device=device, config=config)
                model.train()
                item.update(val_metrics)
                monitor_value = item[monitor_metric]
                if np.isfinite(monitor_value) and monitor_value < best_metric - min_delta:
                    best_metric = monitor_value
                    best_state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            epoch_history.append(item)
            if epoch_callback is not None:
                epoch_callback(item)
            if patience is not None and validation_arrays is not None and stale_epochs >= patience:
                item["early_stop"] = 1.0
                break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
    return epoch_history


def early_stopping_patience(training: dict[str, Any]) -> int | None:
    """Return patience when validation early stopping is enabled."""
    config = training.get("early_stopping")
    if not config:
        return None
    if isinstance(config, bool):
        return int(training.get("patience", 10)) if config else None
    if not isinstance(config, dict) or not config.get("enabled", False):
        return None
    patience = int(config.get("patience", 10))
    return max(1, patience)


def early_stopping_min_delta(training: dict[str, Any]) -> float:
    config = training.get("early_stopping")
    if isinstance(config, dict):
        return float(config.get("min_delta", 0.0))
    return float(training.get("min_delta", 0.0))


def early_stopping_monitor_metric(training: dict[str, Any]) -> str:
    """Return the validation metric key used for best-epoch selection."""
    config = training.get("early_stopping")
    if isinstance(config, dict):
        return str(config.get("metric", "val_rmse"))
    return str(training.get("early_stopping_metric", "val_rmse"))


def evaluate_torch_validation_metrics(model, arrays, device=None, config: dict[str, Any] | None = None) -> dict[str, float]:
    """Evaluate validation metrics including effective generation-period metrics."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    predictions = predict_dataframe("validation", model, arrays, config=config)
    metrics = {f"val_{name}": value for name, value in prediction_frame_metrics(predictions).items()}
    metrics.update({f"val_generation_{name}": value for name, value in generation_period_metrics(predictions).items()})
    return metrics


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
