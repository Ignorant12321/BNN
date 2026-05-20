"""统一加载 run 并评估。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.artifacts.run_io import load_model_artifact
from src.data.pv import load_split_window_arrays_from_config
from src.evaluation.metrics import regression_metrics
from src.evaluation.predictor import predict_dataframe


@dataclass(frozen=True)
class EvaluationResult:
    """单个 run 的统一评估结果。"""

    label: str
    run_dir: Path
    model_name: str
    metrics: dict[str, float]
    predictions: pd.DataFrame


def evaluate_run(label: str, run_dir: str | Path, split: str = "test") -> EvaluationResult:
    """加载 run 的 best 模型，并在指定 split 上统一评估。"""
    run_path = Path(run_dir)
    model, config = load_model_artifact(run_path)
    arrays = load_split_window_arrays_from_config(config)[split]
    predictions = predict_dataframe(label, model, arrays)
    metrics = regression_metrics(
        predictions.pivot(index="sample", columns="horizon", values="mean").to_numpy(),
        predictions.pivot(index="sample", columns="horizon", values="log_var").to_numpy(),
        arrays.target,
    )
    return EvaluationResult(
        label=label,
        run_dir=run_path,
        model_name=str(config.get("model", {}).get("name", run_path.parent.name)),
        metrics={f"{split}_{name}": value for name, value in metrics.items()},
        predictions=predictions,
    )

