"""实验图表绘制。

这些图会自动保存到每次实验的 `figures/` 目录，用于论文结果展示：
loss 曲线、预测区间、不同预测步长误差、区间覆盖率/宽度以及校准曲线。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import horizon_metrics, pinaw, picp
from src.predict import interval_from_mean_std


def plot_loss_curve(train_losses: list[float], val_losses: list[float], path: str | Path) -> None:
    """绘制训练集和验证集 loss 曲线。"""
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_prediction_interval(
    y_true: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    path: str | Path,
    times: np.ndarray | None = None,
    max_points: int = 160,
) -> None:
    """绘制真实值、预测均值和预测区间。

    如果传入 times，则认为调用方已经选好了展示时间段；否则兼容旧逻辑，
    把多步预测结果展平成一条序列并展示前 max_points 个点。
    """
    y = y_true.reshape(-1)[:max_points]
    m = mean.reshape(-1)[:max_points]
    lo = lower.reshape(-1)[:max_points]
    hi = upper.reshape(-1)[:max_points]
    x = np.arange(len(y))
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label="true", linewidth=1)
    plt.plot(x, m, label="mean", linewidth=1)
    plt.fill_between(x, lo, hi, alpha=0.25, label="interval")
    if times is not None and len(times) > 0:
        tick_count = min(6, len(times))
        tick_positions = np.linspace(0, len(times) - 1, tick_count, dtype=int)
        tick_labels = [np.datetime_as_string(np.asarray(times)[i], unit="m").replace("T", "\n") for i in tick_positions]
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel("Target time")
    else:
        plt.xlabel("Step")
    plt.ylabel("AC Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_horizon_rmse(y_true: np.ndarray, mean: np.ndarray, path: str | Path) -> None:
    """绘制不同预测步长上的 RMSE。"""
    rows = horizon_metrics(y_true, mean)
    x = [r["horizon"] for r in rows]
    y = [r["rmse"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.bar(x, y)
    plt.xlabel("Forecast horizon")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_picp_pinaw(y_true: np.ndarray, intervals: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path) -> None:
    """绘制各置信区间的覆盖率和归一化宽度。"""
    labels = list(intervals)
    picps = [picp(y_true, intervals[k][0], intervals[k][1]) for k in labels]
    pinaws = [pinaw(y_true, intervals[k][0], intervals[k][1]) for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, picps, width=width, label="PICP")
    plt.bar(x + width / 2, pinaws, width=width, label="PINAW")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, path: str | Path) -> None:
    """绘制概率校准曲线。

    横轴是名义覆盖率，纵轴是实际覆盖率。曲线越接近对角线，说明模型给出的
    不确定性越可信。
    """
    levels = np.arange(0.1, 1.0, 0.1)
    observed = []
    for level in levels:
        lower, upper = interval_from_mean_std(mean, std, level)
        observed.append(picp(y_true, lower, upper))
    plt.figure(figsize=(5, 5))
    plt.plot(levels, observed, marker="o", label="observed")
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Observed coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_model_prediction_mean(
    predictions_by_model: dict[str, pd.DataFrame],
    path: str | Path,
    max_points: int = 160,
) -> None:
    """把多个模型的一步预测均值画在同一张时间序列图上。"""
    plt.figure(figsize=(10, 4))
    true_plotted = False
    for model_name, predictions in predictions_by_model.items():
        if predictions.empty:
            continue
        horizon = int(predictions["horizon"].min())
        view = predictions[predictions["horizon"] == horizon].sort_values(["sample", "target_time"]).head(max_points)
        x = np.arange(len(view))
        if not true_plotted:
            plt.plot(x, view["y_true"].to_numpy(), label="true", linewidth=1.5, color="black")
            true_plotted = True
        plt.plot(x, view["y_mean"].to_numpy(), label=model_name, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("AC Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_model_horizon_rmse(horizon_rmse: pd.DataFrame, path: str | Path) -> None:
    """把各模型在不同预测步长上的 RMSE 曲线画在同一张图上。"""
    plt.figure(figsize=(7, 4))
    for model_name, group in horizon_rmse.groupby("model", sort=False):
        group = group.sort_values("horizon")
        plt.plot(group["horizon"], group["rmse"], marker="o", label=model_name)
    plt.xlabel("Forecast horizon")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_model_metrics(metrics: pd.DataFrame, path: str | Path, metric_names: list[str] | None = None) -> None:
    """绘制模型整体指标对比柱状图。"""
    if metric_names is None:
        metric_names = [name for name in ["rmse", "mae", "smape", "nrmse"] if name in metrics.columns]
    if not metric_names:
        return

    labels = metrics["model"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(metric_names)
    plt.figure(figsize=(max(7, len(labels) * 1.2), 4))
    for index, metric_name in enumerate(metric_names):
        offset = (index - (len(metric_names) - 1) / 2) * width
        plt.bar(x + offset, metrics[metric_name].astype(float).to_numpy(), width=width, label=metric_name)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
