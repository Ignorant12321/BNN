"""单模型训练流程。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from src.evaluation.metrics import evaluate_arrays
from src.training.torch_trainer import evaluate_torch_model, train_torch_model


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
        )
        metrics = evaluate_train_val(model, arrays_by_split, lambda fitted_model, arrays: evaluate_torch_model(fitted_model, arrays, config=config))
    elif hasattr(model, "fit"):
        model.fit(train_arrays)
        metrics = evaluate_train_val(model, arrays_by_split, evaluate_arrays)
    else:
        metrics = evaluate_train_val(model, arrays_by_split, evaluate_arrays)
    best_epoch = best_epoch_from_history(epoch_history)
    return TrainResult(metrics=metrics, epoch_history=epoch_history, best_epoch=best_epoch)


def evaluate_train_val(model, arrays_by_split: dict[str, Any], evaluator) -> dict[str, float]:
    """分别评估 train/val，训练阶段不碰 test split。"""
    metrics: dict[str, float] = {}
    for split_name in ("train", "val"):
        split_metrics = evaluator(model, arrays_by_split[split_name])
        for metric_name, value in split_metrics.items():
            metrics[f"{split_name}_{metric_name}"] = value
    return metrics


def best_epoch_from_history(epoch_history: list[dict[str, float]]) -> int | None:
    """根据验证 RMSE 或训练 loss 选择最佳 epoch。"""
    if not epoch_history:
        return None
    best = min(epoch_history, key=lambda item: item.get("val_rmse", item["loss"]))
    return int(best["epoch"])
