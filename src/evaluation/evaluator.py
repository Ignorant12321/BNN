"""统一加载训练产物并评估。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.artifacts.run_io import load_model_artifact
from src.data.pv import load_split_window_arrays_from_config
from src.data.scaling import scaler_from_config, transform_window_arrays
from src.evaluation.metrics import generation_period_metrics, regression_metrics
from src.evaluation.predictor import predict_dataframe


@dataclass(frozen=True)
class EvaluationResult:
    """单个训练产物的统一评估结果。"""

    label: str
    run_dir: Path
    model_name: str
    metrics: dict[str, float]
    predictions: pd.DataFrame


def evaluate_run(label: str, run_dir: str | Path, split: str = "test") -> EvaluationResult:
    """加载训练产物的 best 模型，并在指定 split 上统一评估。"""
    run_path = Path(run_dir)
    model, config = load_model_artifact(run_path)
    arrays = load_split_window_arrays_from_config(config)[split]
    scaler = scaler_from_config(config)
    if scaler is not None:
        arrays = transform_window_arrays(arrays, scaler)
    predictions = predict_dataframe(label, model, arrays, config=config)
    metrics = regression_metrics(
        predictions.pivot(index="sample", columns="horizon", values="mean").to_numpy(),
        predictions.pivot(index="sample", columns="horizon", values="log_var").to_numpy(),
        predictions.pivot(index="sample", columns="horizon", values="target").to_numpy(),
    )
    generation_metrics = generation_period_metrics(predictions)
    flattened_metrics = {f"{split}_{name}": value for name, value in metrics.items()}
    flattened_metrics.update({f"{split}_generation_{name}": value for name, value in generation_metrics.items()})
    return EvaluationResult(
        label=label,
        run_dir=run_path,
        model_name=str(config.get("model", {}).get("name", run_path.parent.name)),
        metrics=flattened_metrics,
        predictions=predictions,
    )
