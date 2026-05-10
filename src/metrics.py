"""评价指标。

本模块同时提供点预测指标和概率预测指标：

- MAE / RMSE / nRMSE / sMAPE 衡量预测均值的准确性。
- CRPS / NLL / PICP / PINAW 衡量预测分布和区间质量。
"""

from __future__ import annotations

import math

import numpy as np


def mae(y_true, y_pred) -> float:
    """平均绝对误差。"""
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """均方根误差，对大误差更敏感。"""
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true, y_pred) -> float:
    """归一化 RMSE，用真实值范围做归一化。"""
    y_true, y_pred = _arrays(y_true, y_pred)
    denom = np.nanmax(y_true) - np.nanmin(y_true)
    return float(rmse(y_true, y_pred) / max(denom, 1e-8))


def smape(y_true, y_pred) -> float:
    """对称平均绝对百分比误差。

    分母使用 |true| + |pred|，并设置下界，避免夜间功率接近 0 时除零。
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom))


def gaussian_nll_np(y_true, mean, var) -> float:
    """NumPy 版本高斯负对数似然，用于训练后评估。"""
    y_true, mean, var = _arrays(y_true, mean, var)
    var = np.maximum(var, 1e-8)
    return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (y_true - mean) ** 2 / var)))


def gaussian_crps_np(y_true, mean, std) -> float:
    """Gaussian CRPS，用于评价正态预测分布的整体质量。"""
    y_true, mean, std = _arrays(y_true, mean, std)
    std = np.maximum(std, 1e-8)
    z = (y_true - mean) / std
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    crps = std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def picp(y_true, lower, upper) -> float:
    """Prediction Interval Coverage Probability，预测区间覆盖率。"""
    y_true, lower, upper = _arrays(y_true, lower, upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def pinaw(y_true, lower, upper) -> float:
    """Prediction Interval Normalized Average Width，归一化区间宽度。"""
    y_true, lower, upper = _arrays(y_true, lower, upper)
    width = np.mean(upper - lower)
    data_range = np.nanmax(y_true) - np.nanmin(y_true)
    return float(width / max(data_range, 1e-8))


def horizon_metrics(y_true, y_pred) -> list[dict[str, float]]:
    """逐预测步计算 MAE 和 RMSE。

    返回列表中的 horizon 从 1 开始，对应未来 15 分钟、30 分钟、...、
    4 小时。论文中可用它画 horizon RMSE 曲线。
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    return [
        {"horizon": i + 1, "mae": mae(y_true[:, i], y_pred[:, i]), "rmse": rmse(y_true[:, i], y_pred[:, i])}
        for i in range(y_true.shape[1])
    ]


def _arrays(*items):
    """把输入统一转换为 float 类型 NumPy 数组。"""
    return [np.asarray(item, dtype=float) for item in items]
