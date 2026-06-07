"""单模型训练流程。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from src.evaluation.metrics import evaluate_arrays
from src.training.torch_trainer import early_stopping_monitor_metric, evaluate_torch_model, train_torch_model


@dataclass(frozen=True)
class TrainResult:
    """训练后的过程指标。"""

    metrics: dict[str, float]
    epoch_history: list[dict[str, float]]
    best_epoch: int | None


def train_model(
    model,
    arrays_by_split: dict[str, Any],
    config: dict[str, Any],
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
    validation_metrics_callback: Callable[[Any], dict[str, float]] | None = None,
) -> TrainResult:
    """只使用 train split 拟合，并只评估 train/val 过程指标。"""
    train_arrays = arrays_by_split["train"]
    epoch_history: list[dict[str, float]] = []
    if getattr(model, "is_torch_model", False):
        epoch_history = train_torch_model(
            model,
            train_arrays,
            config,
            epoch_callback=epoch_callback,
            validation_arrays=arrays_by_split.get("val"),
            validation_metrics_callback=validation_metrics_callback,
        )
        metrics = evaluate_train_val(model, arrays_by_split, lambda fitted_model, arrays: evaluate_torch_model(fitted_model, arrays, config=config))
    elif hasattr(model, "fit"):
        model.fit(train_arrays)
        metrics = evaluate_train_val(model, arrays_by_split, evaluate_arrays)
    else:
        metrics = evaluate_train_val(model, arrays_by_split, evaluate_arrays)
    best_epoch = best_epoch_from_history(epoch_history, monitor_metric=training_monitor_metric(config))
    return TrainResult(metrics=metrics, epoch_history=epoch_history, best_epoch=best_epoch)


def evaluate_train_val(model, arrays_by_split: dict[str, Any], evaluator) -> dict[str, float]:
    """分别评估 train/val，训练阶段不碰 test split。"""
    metrics: dict[str, float] = {}
    for split_name in ("train", "val"):
        split_metrics = evaluator(model, arrays_by_split[split_name])
        for metric_name, value in split_metrics.items():
            metrics[f"{split_name}_{metric_name}"] = value
    return metrics


def training_monitor_metric(config: dict[str, Any]) -> str:
    """Return the metric used to identify the best epoch for this training run."""
    return early_stopping_monitor_metric(config.get("training", {}))


def best_epoch_from_history(epoch_history: list[dict[str, float]], monitor_metric: str = "val_rmse") -> int | None:
    """根据配置的验证指标或训练 loss 选择最佳 epoch。"""
    if not epoch_history:
        return None
    best = min(epoch_history, key=lambda item: item.get(monitor_metric, item.get("val_rmse", item["loss"])))
    return int(best["epoch"])
