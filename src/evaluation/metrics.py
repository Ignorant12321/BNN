"""统一评估指标。"""

from __future__ import annotations

import numpy as np


BASE_METRIC_NAMES = (
    "mae",
    "rmse",
    "nmae",
    "nrmse",
    "picp_90",
    "pinaw_90",
    "picp_95",
    "pinaw_95",
)

INTERVAL_Z = {
    "90": 1.6448536269514722,
    "95": 1.959963984540054,
}


def regression_metrics(mean: np.ndarray, log_var: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """计算概率预测指标。"""
    aligned_target = target[: len(mean)]
    error = mean - aligned_target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean((mean - aligned_target) ** 2)))
    scale = normalization_scale(aligned_target)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / scale,
        "nrmse": rmse / scale,
    }
    std = np.sqrt(np.exp(log_var[: len(mean)]))
    for level, z_value in INTERVAL_Z.items():
        lower = mean - z_value * std
        upper = mean + z_value * std
        coverage = np.logical_and(aligned_target >= lower, aligned_target <= upper)
        width = upper - lower
        metrics[f"picp_{level}"] = float(np.mean(coverage))
        metrics[f"pinaw_{level}"] = float(np.mean(width) / scale)
    return metrics


def normalization_scale(target: np.ndarray) -> float:
    """返回归一化指标和 PINAW 共用的目标值尺度。"""
    target_range = float(np.max(target) - np.min(target))
    if target_range > 0:
        return target_range
    mean_abs = float(np.mean(np.abs(target)))
    if mean_abs > 0:
        return mean_abs
    return 1.0


def evaluate_arrays(model, arrays) -> dict[str, float]:
    """用 NumPy 风格模型评估一个 WindowArrays。"""
    mean, log_var = model(arrays.as_batch())
    return regression_metrics(mean, log_var, arrays.target)
