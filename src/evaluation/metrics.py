"""统一评估指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def regression_metrics(
    mean: np.ndarray,
    log_var: np.ndarray,
    target: np.ndarray,
    normalization_scale_value: float | None = None,
) -> dict[str, float]:
    """计算概率预测指标。"""
    aligned_target = target[: len(mean)]
    error = mean - aligned_target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean((mean - aligned_target) ** 2)))
    scale = normalization_scale(aligned_target, normalization_scale_value=normalization_scale_value)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / scale,
        "nrmse": rmse / scale,
    }
    log_var = log_var[: len(mean)]
    if np.isnan(log_var).all():
        for level in INTERVAL_Z:
            metrics[f"picp_{level}"] = float("nan")
            metrics[f"pinaw_{level}"] = float("nan")
        return metrics
    std = np.sqrt(np.exp(log_var))
    for level, z_value in INTERVAL_Z.items():
        lower = mean - z_value * std
        upper = mean + z_value * std
        coverage = np.logical_and(aligned_target >= lower, aligned_target <= upper)
        width = upper - lower
        metrics[f"picp_{level}"] = float(np.mean(coverage))
        metrics[f"pinaw_{level}"] = float(np.mean(width) / scale)
    return metrics


def prediction_frame_metrics(frame: pd.DataFrame, normalization_scale_value: float | None = None) -> dict[str, float]:
    """Calculate regression metrics from flattened prediction rows."""
    if frame.empty:
        return empty_metrics()
    return regression_metrics(
        frame["mean"].to_numpy(dtype=np.float32).reshape(-1, 1),
        frame["log_var"].to_numpy(dtype=np.float32).reshape(-1, 1),
        frame["target"].to_numpy(dtype=np.float32).reshape(-1, 1),
        normalization_scale_value=normalization_scale_value,
    )


def generation_period_metrics(
    frame: pd.DataFrame,
    start: str = "06:00",
    end: str = "18:00",
    normalization_scale_value: float | None = None,
) -> dict[str, float]:
    """Calculate metrics for target times in the effective PV generation period."""
    subset = generation_period_subset(frame, start=start, end=end)
    return prediction_frame_metrics(subset, normalization_scale_value=normalization_scale_value)


def generation_period_subset(frame: pd.DataFrame, start: str = "06:00", end: str = "18:00") -> pd.DataFrame:
    """Return rows whose target_time clock part is between start and end inclusive."""
    if frame.empty or "target_time" not in frame.columns:
        return frame.iloc[0:0].copy()
    target_times = pd.to_datetime(frame["target_time"], errors="coerce")
    clock = target_times.dt.time
    start_time = pd.to_datetime(start, format="%H:%M").time()
    end_time = pd.to_datetime(end, format="%H:%M").time()
    mask = target_times.notna() & (clock >= start_time) & (clock <= end_time)
    return frame.loc[mask].copy()


def empty_metrics() -> dict[str, float]:
    """Return metric placeholders for an empty evaluation subset."""
    return {name: float("nan") for name in BASE_METRIC_NAMES}


def normalization_scale(target: np.ndarray, normalization_scale_value: float | None = None) -> float:
    """返回归一化指标和 PINAW 共用的目标值尺度。"""
    if normalization_scale_value is not None and float(normalization_scale_value) > 0:
        return float(normalization_scale_value)
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


def normalization_scale_from_config(config: dict | None) -> float | None:
    """Return the fixed metric normalization scale saved with a fitted scaler."""
    scaling = (config or {}).get("data", {}).get("scaling", {})
    scaler = scaling.get("scaler") if isinstance(scaling, dict) else None
    normalization = scaler.get("normalization") if isinstance(scaler, dict) else None
    if not isinstance(normalization, dict):
        return None
    scale = normalization.get("scale")
    if scale is None:
        return None
    scale = float(scale)
    return scale if scale > 0 else None
