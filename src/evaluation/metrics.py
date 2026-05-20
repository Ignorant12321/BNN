"""统一评估指标。"""

from __future__ import annotations

import numpy as np


def regression_metrics(mean: np.ndarray, log_var: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """计算概率预测指标。"""
    aligned_target = target[: len(mean)]
    rmse = float(np.sqrt(np.mean((mean - aligned_target) ** 2)))
    nll = float(np.mean(log_var + (aligned_target - mean) ** 2 / np.exp(log_var)))
    return {"rmse": rmse, "nll": nll}


def evaluate_arrays(model, arrays) -> dict[str, float]:
    """用 NumPy 风格模型评估一个 WindowArrays。"""
    mean, log_var = model(arrays.as_batch())
    return regression_metrics(mean, log_var, arrays.target)

