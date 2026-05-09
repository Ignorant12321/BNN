"""训练后评估入口。

`evaluate_predictions` 接收已经反标准化到真实功率尺度的预测结果，并汇总
点预测与概率预测指标。
"""

from __future__ import annotations

import numpy as np

from src.metrics import gaussian_nll_np, mae, nrmse, pinaw, picp, rmse, smape
from src.predict import interval_from_samples


def evaluate_predictions(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    """计算测试集总体指标。

    90% 和 95% 区间来自 MC 样本分位数；NLL 使用 mean 和 std 表示的高斯分布。
    """
    lower90, upper90 = interval_from_samples(samples, 0.90)
    lower95, upper95 = interval_from_samples(samples, 0.95)
    return {
        "mae": mae(y_true, mean),
        "rmse": rmse(y_true, mean),
        "nrmse": nrmse(y_true, mean),
        "smape": smape(y_true, mean),
        "nll": gaussian_nll_np(y_true, mean, std**2),
        "picp_90": picp(y_true, lower90, upper90),
        "pinaw_90": pinaw(y_true, lower90, upper90),
        "picp_95": picp(y_true, lower95, upper95),
        "pinaw_95": pinaw(y_true, lower95, upper95),
    }
