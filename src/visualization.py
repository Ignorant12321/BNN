from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import horizon_metrics, pinaw, picp


def plot_loss_curve(train_losses: list[float], val_losses: list[float], path: str | Path) -> None:
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
    max_points: int = 160,
) -> None:
    y = y_true.reshape(-1)[:max_points]
    m = mean.reshape(-1)[:max_points]
    lo = lower.reshape(-1)[:max_points]
    hi = upper.reshape(-1)[:max_points]
    x = np.arange(len(y))
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label="true", linewidth=1)
    plt.plot(x, m, label="mean", linewidth=1)
    plt.fill_between(x, lo, hi, alpha=0.25, label="interval")
    plt.xlabel("Step")
    plt.ylabel("AC Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_horizon_rmse(y_true: np.ndarray, mean: np.ndarray, path: str | Path) -> None:
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


def plot_calibration_curve(y_true: np.ndarray, samples: np.ndarray, path: str | Path) -> None:
    levels = np.arange(0.1, 1.0, 0.1)
    observed = []
    for level in levels:
        alpha = 1 - level
        lower = np.quantile(samples, alpha / 2, axis=0)
        upper = np.quantile(samples, 1 - alpha / 2, axis=0)
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
