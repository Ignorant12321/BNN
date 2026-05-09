from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    denom = np.nanmax(y_true) - np.nanmin(y_true)
    return float(rmse(y_true, y_pred) / max(denom, 1e-8))


def smape(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom))


def gaussian_nll_np(y_true, mean, var) -> float:
    y_true, mean, var = _arrays(y_true, mean, var)
    var = np.maximum(var, 1e-8)
    return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (y_true - mean) ** 2 / var)))


def picp(y_true, lower, upper) -> float:
    y_true, lower, upper = _arrays(y_true, lower, upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def pinaw(y_true, lower, upper) -> float:
    y_true, lower, upper = _arrays(y_true, lower, upper)
    width = np.mean(upper - lower)
    data_range = np.nanmax(y_true) - np.nanmin(y_true)
    return float(width / max(data_range, 1e-8))


def horizon_metrics(y_true, y_pred) -> list[dict[str, float]]:
    y_true, y_pred = _arrays(y_true, y_pred)
    return [
        {"horizon": i + 1, "mae": mae(y_true[:, i], y_pred[:, i]), "rmse": rmse(y_true[:, i], y_pred[:, i])}
        for i in range(y_true.shape[1])
    ]


def _arrays(*items):
    return [np.asarray(item, dtype=float) for item in items]
