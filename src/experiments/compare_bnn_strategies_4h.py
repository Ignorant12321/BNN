"""Compare 4-hour BNN forecasting strategies: Recursive, Direct, and MIMO.

Usage:
    python -m src.experiments.compare_bnn_strategies_4h
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import deep_update, load_config
from src.data.pv import WindowArrays
from src.data.scaling import (
    attach_fitted_scaler,
    fit_window_scaler,
    inverse_target_prediction,
    inverse_target_values,
    should_scale_torch_training,
    transform_window_arrays_by_split,
)
from src.evaluation.metrics import BASE_METRIC_NAMES, generation_period_metrics, prediction_frame_metrics
from src.evaluation.predictor import predict_arrays, predict_dataframe, prediction_interval_columns
from src.experiments.compare import write_metrics_csv
from src.experiments.train import load_or_make_split_arrays, run_training, set_random_seed
from src.models.registry import build_model
from src.training.trainer import train_model

DEFAULT_BNN_4H_CONFIG = Path("configs/models/bnn/4h.yaml")


@dataclass(frozen=True)
class StrategyRun:
    label: str
    model: str
    run_dir: Path
    metrics: dict[str, float]
    duration_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 4h Recursive, Direct, and MIMO BNN strategies.")
    parser.add_argument("--config", default=str(DEFAULT_BNN_4H_CONFIG), help="Base 4h BNN config used by all strategies.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all freshly trained strategies.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override MC samples for all freshly trained strategies.")
    args = parser.parse_args()

    out_dir = run_strategy_comparison(
        config_path=args.config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        n_samples=args.n_samples,
    )
    print(f"4h BNN strategy comparison written to: {out_dir}")


def run_strategy_comparison(
    config_path: str | Path = DEFAULT_BNN_4H_CONFIG,
    output_dir: str | Path = "outputs",
    epochs: int | None = None,
    n_samples: int | None = None,
) -> Path:
    """Train all three 4h strategies and write a compact comparison table."""
    compare_dir = Path(output_dir) / "comparisons" / f"bnn_4h_strategies_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    predictions_dir = compare_dir / "predictions"
    compare_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    base_config = strategy_config(load_config(config_path), epochs=epochs, n_samples=n_samples)

    direct_config = copy.deepcopy(base_config)
    direct = run_direct_strategy(direct_config, compare_dir / "direct")
    direct_predictions = pd.read_csv(direct.run_dir / "predictions" / "test.csv")
    direct_predictions.to_csv(predictions_dir / "Direct.csv", index=False)

    mimo_config = copy.deepcopy(base_config)
    mimo_config["run_dir"] = str(compare_dir / "mimo")
    mimo_started = time.perf_counter()
    mimo_run_dir = run_training(mimo_config, note="MIMO 4h strategy")
    mimo_duration = time.perf_counter() - mimo_started
    mimo_metrics = read_split_metrics(mimo_run_dir / "metrics.csv")
    pd.read_csv(mimo_run_dir / "predictions" / "test.csv").to_csv(predictions_dir / "MIMO.csv", index=False)

    recursive_config = copy.deepcopy(base_config)
    recursive = run_recursive_strategy(recursive_config, compare_dir / "recursive")
    pd.read_csv(recursive.run_dir / "predictions" / "test.csv").to_csv(predictions_dir / "Recursive.csv", index=False)

    rows = [
        strategy_row(direct),
        strategy_row(StrategyRun("MIMO", str(mimo_config.get("model", {}).get("name", "improved_bnn")), mimo_run_dir, mimo_metrics, mimo_duration)),
        strategy_row(recursive),
    ]
    write_metrics_csv(compare_dir / "model_metrics.csv", rows)
    write_markdown_summary(compare_dir / "summary.md", rows)
    return compare_dir


def strategy_config(config: dict[str, Any], epochs: int | None = None, n_samples: int | None = None) -> dict[str, Any]:
    """Return a config copy with optional runtime overrides."""
    config = copy.deepcopy(config)
    updates: dict[str, Any] = {}
    if epochs is not None:
        updates.setdefault("training", {})["epochs"] = int(epochs)
    if n_samples is not None:
        updates.setdefault("evaluation", {})["n_samples"] = int(n_samples)
    if updates:
        config = deep_update(config, updates)
    return config


def run_direct_strategy(config: dict[str, Any], run_dir: Path) -> StrategyRun:
    """Train one one-step BNN per horizon and combine the predictions."""
    started = time.perf_counter()
    set_random_seed(int(config.get("seed", 42)))
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays_by_split = load_or_make_split_arrays(config)
    model_probe = build_model(config)
    if getattr(model_probe, "is_torch_model", False) and should_scale_torch_training(config):
        scaler = fit_window_scaler(arrays_by_split)
        config = attach_fitted_scaler(config, scaler)
        arrays_by_split = transform_window_arrays_by_split(arrays_by_split, scaler)

    horizon = int(config["data"]["horizon"])
    test_frames = []
    for step_index in range(horizon):
        step_config = make_direct_step_config(config, step_index)
        step_arrays = {split_name: slice_step_arrays(arrays, step_index) for split_name, arrays in arrays_by_split.items()}
        model = build_model(step_config)
        train_model(model, step_arrays, step_config)
        test_frames.append(predict_dataframe(f"Direct-step-{step_index + 1}", model, step_arrays["test"], config=step_config))

    predictions = combine_direct_prediction_frames(test_frames, label="Direct")
    metrics = {f"test_{name}": value for name, value in prediction_frame_metrics(predictions).items()}
    metrics.update({f"test_generation_{name}": value for name, value in generation_period_metrics(predictions).items()})
    duration_seconds = time.perf_counter() - started

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_dir / "test.csv", index=False)
    write_direct_log(run_dir / "train.log", horizon, duration_seconds, metrics)
    write_direct_metrics(run_dir / "metrics.csv", metrics)
    return StrategyRun("Direct", "improved_bnn_direct", run_dir, metrics, duration_seconds)


def run_recursive_strategy(config: dict[str, Any], run_dir: Path) -> StrategyRun:
    """Train one one-step BNN and roll its prediction forward to the full 4h horizon."""
    started = time.perf_counter()
    set_random_seed(int(config.get("seed", 42)))
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays_by_split = load_or_make_split_arrays(config)
    model_probe = build_model(config)
    if getattr(model_probe, "is_torch_model", False) and should_scale_torch_training(config):
        scaler = fit_window_scaler(arrays_by_split)
        config = attach_fitted_scaler(config, scaler)
        arrays_by_split = transform_window_arrays_by_split(arrays_by_split, scaler)

    step_config = make_recursive_step_config(config)
    step_arrays = {split_name: slice_step_arrays(arrays, 0) for split_name, arrays in arrays_by_split.items()}
    model = build_model(step_config)
    train_model(model, step_arrays, step_config)
    predictions = recursive_prediction_frame("Recursive", model, arrays_by_split["test"], config=step_config)
    metrics = {f"test_{name}": value for name, value in prediction_frame_metrics(predictions).items()}
    metrics.update({f"test_generation_{name}": value for name, value in generation_period_metrics(predictions).items()})
    duration_seconds = time.perf_counter() - started

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_dir / "test.csv", index=False)
    write_direct_log(run_dir / "train.log", int(config["data"]["horizon"]), duration_seconds, metrics, title="Recursive 4h BNN Strategy")
    write_direct_metrics(run_dir / "metrics.csv", metrics)
    return StrategyRun("Recursive", "improved_bnn_recursive", run_dir, metrics, duration_seconds)


def make_direct_step_config(config: dict[str, Any], step_index: int) -> dict[str, Any]:
    """Build a one-step config for a specific future horizon index."""
    step_config = deep_update(copy.deepcopy(config), {"data": {"horizon": 1}})
    step_config["run_dir"] = str(Path(config.get("run_dir", "")) / f"step-{step_index + 1:02d}") if config.get("run_dir") else ""
    return step_config


def make_recursive_step_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build the one-step config used by the recursive strategy."""
    step_config = deep_update(copy.deepcopy(config), {"data": {"horizon": 1}})
    step_config["run_dir"] = str(Path(config.get("run_dir", "")) / "recursive-step") if config.get("run_dir") else ""
    return step_config


def slice_step_arrays(arrays: WindowArrays, step_index: int) -> WindowArrays:
    """Select one forecast step while keeping the same historical and direct inputs."""
    target_time = None
    if arrays.target_time is not None:
        target_time = arrays.target_time[:, step_index : step_index + 1]
    return WindowArrays(
        history=arrays.history,
        weather=arrays.weather[:, step_index : step_index + 1, :],
        direct=arrays.direct,
        target=arrays.target[:, step_index : step_index + 1],
        target_time=target_time,
    )


def combine_direct_prediction_frames(frames: list[pd.DataFrame], label: str = "Direct") -> pd.DataFrame:
    """Combine one-step prediction frames into a multi-horizon 4h prediction table."""
    combined = []
    for horizon_index, frame in enumerate(frames):
        step_frame = frame.copy()
        step_frame["label"] = label
        step_frame["horizon"] = horizon_index
        combined.append(step_frame)
    if not combined:
        return pd.DataFrame()
    result = pd.concat(combined, ignore_index=True)
    return result.sort_values(["sample", "horizon"]).reset_index(drop=True)


def recursive_prediction_frame(label: str, model, arrays: WindowArrays, config: dict[str, Any]) -> pd.DataFrame:
    """Roll a trained one-step model forward and return a multi-horizon prediction table."""
    horizon = int(arrays.target.shape[1])
    history = arrays.history.copy()
    direct = arrays.direct.copy()
    mean_steps = []
    log_var_steps = []
    for step_index in range(horizon):
        step_arrays = WindowArrays(
            history=history,
            weather=arrays.weather[:, step_index : step_index + 1, :],
            direct=direct,
            target=arrays.target[:, step_index : step_index + 1],
            target_time=None if arrays.target_time is None else arrays.target_time[:, step_index : step_index + 1],
        )
        mean, log_var = predict_arrays(model, step_arrays, config=config)
        step_mean = mean[:, :1].astype(np.float32)
        mean_steps.append(step_mean)
        log_var_steps.append(log_var[:, :1].astype(np.float32))
        direct = roll_direct_input(direct, step_mean)
        history = roll_history_input(history, step_mean)

    mean_scaled = np.concatenate(mean_steps, axis=1)
    log_var_scaled = np.concatenate(log_var_steps, axis=1)
    scaler = config.get("data", {}).get("scaling", {}).get("scaler")
    mean, log_var = inverse_target_prediction(mean_scaled, log_var_scaled, scaler)
    target = inverse_target_values(arrays.target[: len(mean)], scaler)
    return prediction_frame_from_arrays(label, mean, log_var, target, arrays.target_time)


def roll_direct_input(direct: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Feed the previous one-step prediction back into the direct AC_POWER input."""
    updated = direct.copy()
    if updated.shape[1] > 0:
        updated[:, 0] = prediction[:, 0]
    return updated


def roll_history_input(history: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Append the previous one-step prediction to one-feature history windows."""
    if history.shape[1] == 0 or history.shape[2] != 1:
        return history
    return np.concatenate([history[:, 1:, :], prediction[:, None, :]], axis=1).astype(np.float32)


def prediction_frame_from_arrays(
    label: str,
    mean: np.ndarray,
    log_var: np.ndarray,
    target: np.ndarray,
    target_time: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    for sample_index in range(len(mean)):
        for horizon_index in range(mean.shape[1]):
            interval = prediction_interval_columns(float(mean[sample_index, horizon_index]), float(log_var[sample_index, horizon_index]))
            row = {
                "label": label,
                "sample": sample_index,
                "horizon": horizon_index,
                "target": float(target[sample_index, horizon_index]),
                "mean": float(mean[sample_index, horizon_index]),
                "log_var": float(log_var[sample_index, horizon_index]),
                **interval,
            }
            if target_time is not None:
                row["target_time"] = str(target_time[sample_index, horizon_index])
            rows.append(row)
    return pd.DataFrame(rows)


def read_split_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            metrics[f"{row['split']}_{row['metric']}"] = float(row["value"])
    return metrics


def strategy_row(run: StrategyRun) -> dict[str, str]:
    row = {
        "label": run.label,
        "model": run.model,
        "run_dir": str(run.run_dir),
        "duration_seconds": f"{run.duration_seconds:.3f}",
    }
    for split_name in ("test", "test_generation"):
        for metric_name in BASE_METRIC_NAMES:
            key = f"{split_name}_{metric_name}"
            if key in run.metrics:
                row[key] = str(run.metrics[key])
    return row


def write_direct_log(
    path: Path,
    horizon: int,
    duration_seconds: float,
    metrics: dict[str, float],
    title: str = "Direct 4h BNN Strategy",
) -> None:
    lines = [
        title,
        "-" * 48,
        f"Horizon models : {horizon}",
        f"Duration       : {duration_seconds:.3f} s",
    ]
    for name in ("test_mae", "test_rmse", "test_nmae", "test_nrmse"):
        lines.append(f"{name:<14} : {metrics[name]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_direct_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            split, metric = key.rsplit("_", 1)
            if metric in {"90", "95"}:
                split, previous = split.rsplit("_", 1)
                metric = f"{previous}_{metric}"
            writer.writerow({"split": split, "metric": metric, "value": value})


def write_markdown_summary(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["label", "test_mae", "test_rmse", "test_nmae", "test_nrmse", "duration_seconds", "run_dir"]
    lines = [
        "# 4h BNN Strategy Comparison",
        "",
        "| Strategy | Test MAE | Test RMSE | Test NMAE | Test NRMSE | Duration (s) | Run Dir |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        values = [row.get(column, "") for column in columns]
        lines.append(f"| {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {values[5]} | `{values[6]}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
