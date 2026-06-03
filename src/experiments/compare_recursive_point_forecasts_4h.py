"""Compare recursive 4-hour point forecasts for BNN, MLP, 1D-CNN, and LSTM.

Usage:
    python -m src.experiments.compare_recursive_point_forecasts_4h
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.artifacts.run_io import save_config, write_run_note
from src.config import deep_update, load_config
from src.data.scaling import (
    attach_fitted_scaler,
    fit_window_scaler,
    should_scale_torch_training,
    transform_window_arrays_by_split,
)
from src.evaluation.metrics import generation_period_metrics, prediction_frame_metrics
from src.evaluation.plots import write_comparison_loss_png, write_prediction_window_metrics_csv, write_prediction_window_pngs
from src.experiments.compare import write_metrics_csv
from src.experiments.compare_bnn_strategies_4h import make_recursive_step_config, recursive_prediction_frame, slice_step_arrays
from src.experiments.train import load_or_make_split_arrays, set_random_seed, write_epoch_history, write_split_metrics
from src.models.registry import build_model
from src.training.trainer import train_model


DEFAULT_RECURSIVE_POINT_CONFIGS = (
    ("BNN", Path("configs/models/bnn/pv_usibnn_recursive_4h.yaml")),
    ("MLP", Path("configs/models/mlp/recursive_4h.yaml")),
    ("1D-CNN", Path("configs/models/cnn/recursive_4h.yaml")),
    ("LSTM", Path("configs/models/lstm/recursive_4h.yaml")),
)

POINT_METRIC_NAMES = (
    "test_mae",
    "test_rmse",
    "test_nmae",
    "test_nrmse",
    "test_generation_mae",
    "test_generation_rmse",
    "test_generation_nmae",
    "test_generation_nrmse",
)


@dataclass(frozen=True)
class RecursivePointForecastSpec:
    label: str
    config_path: Path
    recursive: bool = True


@dataclass(frozen=True)
class RecursivePointForecastRun:
    label: str
    model: str
    run_dir: Path
    predictions: pd.DataFrame
    metrics: dict[str, float]
    epoch_history: list[dict[str, float]]
    duration_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recursive 4h point forecasts across BNN and common baselines.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for every one-step model.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override BNN MC samples.")
    parser.add_argument("--config", action="append", default=[], help="Optional config path; repeat to override the default four models.")
    args = parser.parse_args()

    out_dir = run_recursive_point_forecast_comparison(
        output_dir=args.output_dir,
        epochs=args.epochs,
        n_samples=args.n_samples,
        configs=args.config or None,
    )
    print(f"Recursive 4h point forecast comparison written to: {out_dir}")


def recursive_point_forecast_specs(configs: list[str | Path] | tuple[str | Path, ...] | None = None) -> list[RecursivePointForecastSpec]:
    """Return the recursive model list used by the point-forecast comparison."""
    if configs is None:
        return [RecursivePointForecastSpec(label, path) for label, path in DEFAULT_RECURSIVE_POINT_CONFIGS]
    default_labels = [label for label, _path in DEFAULT_RECURSIVE_POINT_CONFIGS]
    specs = []
    for index, config_path in enumerate(configs):
        label = default_labels[index] if index < len(default_labels) else Path(config_path).stem
        specs.append(RecursivePointForecastSpec(label, Path(config_path)))
    return specs


def run_recursive_point_forecast_comparison(
    output_dir: str | Path = "outputs",
    epochs: int | None = None,
    n_samples: int | None = None,
    configs: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path:
    """Train each one-step model, roll it recursively to 4h, and write comparison artifacts."""
    compare_dir = Path(output_dir) / "comparisons" / f"recursive_point_forecasts_4h_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    compare_dir.mkdir(parents=True, exist_ok=True)
    write_run_note(compare_dir, "Recursive 4h point forecast comparison")

    specs = recursive_point_forecast_specs(configs)
    runs = [
        run_recursive_point_forecast(spec, compare_dir / "runs" / safe_filename(spec.label), epochs=epochs, n_samples=n_samples)
        for spec in specs
    ]
    compare_config = {
        "name": "recursive_point_forecasts_4h",
        "output_dir": str(output_dir),
        "runs": [{"label": spec.label, "config": str(spec.config_path), "recursive": spec.recursive} for spec in specs],
    }
    if epochs is not None:
        compare_config["epochs"] = int(epochs)
    if n_samples is not None:
        compare_config["n_samples"] = int(n_samples)
    write_point_forecast_artifacts(compare_dir, runs, compare_config=compare_config)
    return compare_dir


def run_recursive_point_forecast(
    spec: RecursivePointForecastSpec,
    run_dir: Path,
    epochs: int | None = None,
    n_samples: int | None = None,
) -> RecursivePointForecastRun:
    """Train a one-step model from a 4h config and recursively roll it over the test horizon."""
    started = time.perf_counter()
    config = recursive_runtime_config(load_config(spec.config_path), epochs=epochs, n_samples=n_samples)
    set_random_seed(int(config.get("seed", 42)))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    arrays_by_split = load_or_make_split_arrays(config)
    model_probe = build_model(config)
    if getattr(model_probe, "is_torch_model", False) and should_scale_torch_training(config):
        scaler = fit_window_scaler(arrays_by_split)
        config = attach_fitted_scaler(config, scaler)
        arrays_by_split = transform_window_arrays_by_split(arrays_by_split, scaler)

    step_config = make_recursive_step_config(config)
    step_arrays = {split_name: slice_step_arrays(arrays, 0) for split_name, arrays in arrays_by_split.items()}
    model = build_model(step_config)
    train_result = train_model(model, step_arrays, step_config)
    epoch_history = list(getattr(train_result, "epoch_history", train_result if isinstance(train_result, list) else []))

    predictions = recursive_prediction_frame(spec.label, model, arrays_by_split["test"], config=step_config)
    metrics = point_metrics(predictions)
    duration_seconds = time.perf_counter() - started

    save_config(step_config, run_dir / "config.yaml")
    write_epoch_history(run_dir / "epoch_history.csv", epoch_history)
    write_split_metrics(run_dir / "metrics.csv", metrics)
    predictions.to_csv(run_dir / "predictions" / "test.csv", index=False)

    return RecursivePointForecastRun(
        label=spec.label,
        model=str(step_config.get("model", {}).get("name", spec.label)),
        run_dir=run_dir,
        predictions=predictions,
        metrics=metrics,
        epoch_history=epoch_history,
        duration_seconds=duration_seconds,
    )


def recursive_runtime_config(config: dict[str, Any], epochs: int | None = None, n_samples: int | None = None) -> dict[str, Any]:
    """Apply runtime CLI overrides without changing the source YAML."""
    config = copy.deepcopy(config)
    updates: dict[str, Any] = {}
    if epochs is not None:
        updates.setdefault("training", {})["epochs"] = int(epochs)
    if n_samples is not None:
        updates.setdefault("evaluation", {})["n_samples"] = int(n_samples)
    if updates:
        config = deep_update(config, updates)
    return config


def point_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate point-forecast metrics only, excluding interval coverage/width."""
    metrics = {f"test_{name}": value for name, value in prediction_frame_metrics(predictions).items()}
    metrics.update({f"test_generation_{name}": value for name, value in generation_period_metrics(predictions).items()})
    return {name: metrics[name] for name in POINT_METRIC_NAMES}


def write_point_forecast_artifacts(compare_dir: Path, runs: list[RecursivePointForecastRun], compare_config: dict[str, Any]) -> None:
    """Write metrics, predictions, loss curves, prediction windows, and metric PNGs."""
    predictions_dir = compare_dir / "predictions"
    figures_dir = compare_dir / "figures"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_config(compare_config, compare_dir / "compare_config.yaml")

    rows = []
    frames = []
    histories = []
    for run in runs:
        run.predictions.to_csv(predictions_dir / f"{safe_filename(run.label)}.csv", index=False)
        row = {"label": run.label, "model": run.model, "run_dir": str(run.run_dir)}
        row.update({name: str(run.metrics[name]) for name in POINT_METRIC_NAMES})
        row["duration_seconds"] = str(run.duration_seconds)
        rows.append(row)
        frames.append(run.predictions)
        histories.append({"label": run.label, "history": run.epoch_history})

    write_metrics_csv(compare_dir / "model_metrics.csv", rows)
    write_markdown_summary(compare_dir / "summary.md", rows)
    write_metric_pngs(rows, figures_dir)
    write_prediction_window_pngs(frames, figures_dir)
    if frames:
        write_prediction_window_metrics_csv(pd.concat(frames, ignore_index=True), figures_dir / "prediction_window_metrics.csv")
    write_comparison_loss_png(histories, figures_dir / "loss_curves.png")


def write_metric_pngs(rows: list[dict[str, str]], figures_dir: Path) -> None:
    """Write one bar chart for each point metric in the comparison table."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["label"] for row in rows]
    for metric_name in POINT_METRIC_NAMES:
        values = [float(row[metric_name]) for row in rows]
        fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=140)
        ax.bar(labels, values, color=["#2563eb", "#dc2626", "#16a34a", "#9333ea"][: len(labels)])
        ax.set_title(metric_name)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(figures_dir / f"metrics_{metric_name}.png")
        plt.close(fig)


def write_markdown_summary(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a compact Markdown summary focused on point-prediction metrics."""
    fields = ["label", "model", *POINT_METRIC_NAMES, "duration_seconds"]
    lines = ["# Recursive 4h Point Forecast Comparison", "", "|" + "|".join(fields) + "|", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("|" + "|".join(row.get(field, "") for field in fields) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_filename(value: str) -> str:
    """Return a readable filename-safe label."""
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "run"


if __name__ == "__main__":
    main()
