"""Train and evaluate the recursive 4-hour BNN strategy.

Usage:
    python -m src.experiments.train_bnn_recursive_4h
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.artifacts.manifest import build_manifest
from src.artifacts.run_io import save_config, save_manifest, save_model_artifact, write_run_note
from src.config import deep_update, load_config
from src.data.scaling import (
    attach_fitted_scaler,
    fit_window_scaler,
    should_scale_torch_training,
    transform_window_arrays_by_split,
)
from src.evaluation.metrics import generation_period_metrics, prediction_frame_metrics
from src.evaluation.plots import write_prediction_window_metrics_csv, write_prediction_window_pngs, write_training_loss_png
from src.experiments.compare_bnn_strategies_4h import (
    make_recursive_step_config,
    recursive_prediction_frame,
    slice_step_arrays,
)
from src.experiments.train import (
    format_metric,
    print_epoch_progress,
    print_section,
    print_training_parameters,
    print_training_process_end,
    print_training_process_start,
    load_or_make_split_arrays,
    set_random_seed,
    write_epoch_history,
    write_split_metrics,
)
from src.models.registry import build_model
from src.training.trainer import train_model

DEFAULT_RECURSIVE_BNN_4H_CONFIG = Path("configs/models/bnn/recursive_4h.yaml")


@dataclass(frozen=True)
class RecursiveExperimentResult:
    run_dir: Path
    metrics: dict[str, float]
    predictions: pd.DataFrame
    epoch_history: list[dict[str, float]]
    model_path: Path
    best_epoch: int | None = None
    duration_seconds: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the recursive 4h BNN strategy.")
    parser.add_argument("--config", default=str(DEFAULT_RECURSIVE_BNN_4H_CONFIG))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for this run.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override MC samples for this run.")
    args = parser.parse_args()

    out_dir = run_recursive_four_hour_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        n_samples=args.n_samples,
    )
    print(f"Recursive 4h BNN results written to: {out_dir}")


def run_recursive_four_hour_experiment(
    config_path: str | Path = DEFAULT_RECURSIVE_BNN_4H_CONFIG,
    output_dir: str | Path = "outputs",
    epochs: int | None = None,
    n_samples: int | None = None,
) -> Path:
    """Train the recursive 4h BNN and write a training-style run directory."""
    config = recursive_runtime_config(load_config(config_path), epochs=epochs, n_samples=n_samples)
    run_dir = make_recursive_train_run_dir(output_dir)
    result = run_recursive_training(config, run_dir=run_dir, note="Recursive 4h BNN strategy")
    return result.run_dir


def make_recursive_train_run_dir(output_dir: str | Path = "outputs") -> Path:
    """Return the canonical training run directory for recursive 4h BNN."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(output_dir) / "train" / "improved_bnn_recursive" / timestamp


def run_recursive_training(config: dict[str, Any], run_dir: Path, note: str | None = None) -> RecursiveExperimentResult:
    """Train one-step BNN and save recursive 4h outputs like a normal training run."""
    started_at = datetime.now()
    started_timer = time.perf_counter()
    set_random_seed(int(config.get("seed", 42)))
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    figures_dir = run_dir / "figures"
    predictions_dir = run_dir / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    write_run_note(run_dir, note)

    arrays_by_split = load_or_make_split_arrays(config)
    model_probe = build_model(config)
    if getattr(model_probe, "is_torch_model", False) and should_scale_torch_training(config):
        scaler = fit_window_scaler(arrays_by_split)
        config = attach_fitted_scaler(config, scaler)
        arrays_by_split = transform_window_arrays_by_split(arrays_by_split, scaler)
    split_sizes = {split_name: len(arrays.target) for split_name, arrays in arrays_by_split.items()}

    step_config = make_recursive_step_config(config)
    step_arrays = {split_name: slice_step_arrays(arrays, 0) for split_name, arrays in arrays_by_split.items()}
    model = build_model(step_config)
    print_training_parameters({**step_config, "model": {"name": "improved_bnn_recursive"}}, run_dir, model, started_at, split_sizes)
    fit_started_at = datetime.now()
    fit_started_timer = time.perf_counter()
    print_training_process_start(fit_started_at)
    train_result = train_model(model, step_arrays, step_config, epoch_callback=print_epoch_progress)
    fit_ended_at = datetime.now()
    print_training_process_end(fit_ended_at, time.perf_counter() - fit_started_timer, train_result.epoch_history)
    val_predictions = recursive_prediction_frame("Recursive", model, arrays_by_split["val"], config=step_config)
    predictions = recursive_prediction_frame("Recursive", model, arrays_by_split["test"], config=step_config)
    metrics = dict(train_result.metrics)
    metrics.update(recursive_prediction_metrics("val", val_predictions))
    metrics.update(recursive_prediction_metrics("test", predictions))

    model_path = save_model_artifact(model, step_config, models_dir, stem="best")
    ended_at = datetime.now()
    duration_seconds = time.perf_counter() - started_timer
    result = RecursiveExperimentResult(
        run_dir=run_dir,
        metrics=metrics,
        predictions=predictions,
        epoch_history=train_result.epoch_history,
        model_path=model_path,
        best_epoch=train_result.best_epoch,
        duration_seconds=duration_seconds,
    )
    write_recursive_outputs(run_dir, result)
    saved_config = recursive_saved_config(config, step_config)
    save_config(saved_config, run_dir / "config.yaml")
    save_manifest(
        build_manifest(
            run_dir=run_dir,
            config={**config, "model": {"name": "improved_bnn_recursive"}},
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            split_sizes=split_sizes,
            model_path=model_path,
            best_epoch=train_result.best_epoch,
        ),
        run_dir / "manifest.json",
    )
    write_recursive_log(run_dir / "train.log", result)
    print_recursive_training_results(result)
    return result


def recursive_saved_config(forecast_config: dict[str, Any], step_config: dict[str, Any]) -> dict[str, Any]:
    """Save a loadable one-step model config while recording the 4h recursive strategy."""
    saved = copy.deepcopy(step_config)
    strategy = copy.deepcopy(forecast_config.get("strategy", {}))
    strategy.setdefault("name", "recursive")
    strategy.setdefault("train_horizon", int(step_config["data"]["horizon"]))
    strategy.setdefault("forecast_horizon", int(forecast_config["data"]["horizon"]))
    saved["strategy"] = strategy
    return saved


def recursive_runtime_config(config: dict, epochs: int | None = None, n_samples: int | None = None) -> dict:
    """Return a config copy with optional CLI overrides."""
    result = copy.deepcopy(config)
    updates: dict = {}
    if epochs is not None:
        updates.setdefault("training", {})["epochs"] = int(epochs)
    if n_samples is not None:
        updates.setdefault("evaluation", {})["n_samples"] = int(n_samples)
    if updates:
        result = deep_update(result, updates)
    return result


def recursive_prediction_metrics(split_name: str, predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate full-period and generation-period metrics for recursive predictions."""
    metrics = {f"{split_name}_{name}": value for name, value in prediction_frame_metrics(predictions).items()}
    metrics.update({f"{split_name}_generation_{name}": value for name, value in generation_period_metrics(predictions).items()})
    return metrics


def write_recursive_outputs(output_dir: Path, run: RecursiveExperimentResult) -> None:
    """Write recursive predictions, metrics, loss history, and figures."""
    predictions_dir = output_dir / "predictions"
    figures_dir = output_dir / "figures"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    run.predictions.to_csv(predictions_dir / "test.csv", index=False)
    write_split_metrics(output_dir / "metrics.csv", run.metrics)
    write_epoch_history(output_dir / "epoch_history.csv", run.epoch_history)
    write_prediction_window_pngs([run.predictions], figures_dir)
    write_prediction_window_metrics_csv(run.predictions, figures_dir / "prediction_window_metrics.csv")
    write_training_loss_png(run.epoch_history, run.metrics, figures_dir / "loss_curve.png", best_epoch=run.best_epoch)


def write_recursive_log(path: Path, run: RecursiveExperimentResult) -> None:
    lines = [
        "Recursive 4h BNN Training",
        "-" * 48,
        f"Duration       : {run.duration_seconds:.3f} s",
        f"Best Epoch     : {run.best_epoch or ''}",
        f"Model File     : {run.model_path}",
    ]
    for name in ("test_mae", "test_rmse", "test_nmae", "test_nrmse"):
        if name in run.metrics:
            lines.append(f"{name:<14} : {run.metrics[name]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_recursive_training_results(run: RecursiveExperimentResult) -> None:
    """Print the same high-level result section used by normal training."""
    print_section(
        "Training Results",
        [
            ("Duration", f"{run.duration_seconds:.2f} s"),
            ("Best Epoch", "" if run.best_epoch is None else str(run.best_epoch)),
            ("Test MAE", format_metric(run.metrics.get("test_mae"))),
            ("Test RMSE", format_metric(run.metrics.get("test_rmse"))),
            ("Test NMAE", format_metric(run.metrics.get("test_nmae"))),
            ("Test NRMSE", format_metric(run.metrics.get("test_nrmse"))),
            ("Output", str(run.run_dir)),
            ("Model File", str(run.model_path)),
        ],
    )
    print("Result files written.", flush=True)


if __name__ == "__main__":
    main()
