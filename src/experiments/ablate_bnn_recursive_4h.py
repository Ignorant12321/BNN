"""Run Recursive 4h BNN input-branch ablations.

Usage:
    python -m src.experiments.ablate_bnn_recursive_4h
    python -m src.experiments.ablate_bnn_recursive_4h --epochs 50 --n-samples 30
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

from src.artifacts.manifest import build_manifest
from src.artifacts.run_io import save_config, save_manifest, save_model_artifact, write_run_note
from src.config import load_config
from src.data.pv import DEFAULT_FEATURE_COLUMNS, WindowArrays
from src.data.scaling import (
    attach_fitted_scaler,
    fit_window_scaler,
    should_scale_torch_training,
    transform_window_arrays_by_split,
)
from src.evaluation.metrics import BASE_METRIC_NAMES
from src.experiments.compare_bnn_strategies_4h import (
    make_recursive_step_config,
    recursive_forecast_config,
    recursive_prediction_frame,
    slice_step_arrays,
)
from src.experiments.train import (
    load_or_make_split_arrays,
    print_epoch_progress,
    print_training_parameters,
    print_training_process_end,
    print_training_process_start,
    set_random_seed,
    write_epoch_history,
    write_split_metrics,
)
from src.experiments.train_bnn_recursive_4h import (
    DEFAULT_RECURSIVE_BNN_4H_CONFIG,
    RecursiveExperimentResult,
    recursive_prediction_metrics,
    recursive_strategy_model_name,
    recursive_runtime_config,
    run_recursive_training,
    write_recursive_outputs,
)
from src.models.registry import build_model
from src.training.trainer import train_model


@dataclass(frozen=True)
class AblationSpec:
    label: str
    remove_feature_group: str | None


ABLATIONS = (
    AblationSpec("baseline", None),
    AblationSpec("no_history", "history"),
    AblationSpec("no_weather", "weather"),
    AblationSpec("no_direct", "direct"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Recursive 4h BNN branch ablations.")
    parser.add_argument("--config", default=str(DEFAULT_RECURSIVE_BNN_4H_CONFIG))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all ablation runs.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override MC samples for all ablation runs.")
    args = parser.parse_args()

    out_dir = run_recursive_four_hour_ablation(
        config_path=args.config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        n_samples=args.n_samples,
    )
    print(f"Recursive 4h BNN ablation written to: {out_dir}")


def run_recursive_four_hour_ablation(
    config_path: str | Path = DEFAULT_RECURSIVE_BNN_4H_CONFIG,
    output_dir: str | Path = "outputs",
    epochs: int | None = None,
    n_samples: int | None = None,
) -> Path:
    """Train baseline plus three one-factor Recursive 4h input ablations."""
    compare_dir = Path(output_dir) / "comparisons" / f"bnn_recursive_4h_ablation_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    base_config = recursive_runtime_config(load_config(config_path), epochs=epochs, n_samples=n_samples)
    rows = []
    for spec in ABLATIONS:
        config = ablation_config(base_config, spec)
        run_dir = compare_dir / spec.label
        result = run_ablation_training(config, run_dir=run_dir, note=f"Recursive 4h BNN ablation: {spec.label}")
        rows.append(ablation_row(spec.label, result))

    write_ablation_metrics(compare_dir / "model_metrics.csv", rows)
    write_ablation_summary(compare_dir / "summary.md", rows)
    return compare_dir


def ablation_config(config: dict[str, Any], spec: AblationSpec) -> dict[str, Any]:
    """Return a Recursive 4h config with one input feature group removed."""
    result = copy.deepcopy(config)
    result.setdefault("data", {})
    result["data"]["features"] = feature_groups(result)
    if spec.remove_feature_group is not None and not uses_zero_feature_ablation(result):
        result["data"]["features"][spec.remove_feature_group] = []
    result.setdefault("strategy", {})
    result["strategy"]["ablation"] = spec.label
    if spec.remove_feature_group is not None and uses_zero_feature_ablation(result):
        result["strategy"]["zero_feature_group"] = spec.remove_feature_group
    return result


def uses_zero_feature_ablation(config: dict[str, Any]) -> bool:
    """Return whether ablation should preserve feature dimensions and zero inputs."""
    return str(config.get("model", {}).get("name", "")) == "pv_usibnn_recursive"


def run_ablation_training(config: dict[str, Any], run_dir: Path, note: str | None = None) -> RecursiveExperimentResult:
    """Run one ablation with model-compatible feature handling."""
    zero_group = config.get("strategy", {}).get("zero_feature_group")
    if zero_group is None:
        return run_recursive_training(config, run_dir=run_dir, note=note)
    return run_zero_feature_recursive_training(config, run_dir=run_dir, zero_group=str(zero_group), note=note)


def feature_groups(config: dict[str, Any]) -> dict[str, Any]:
    """Return explicit feature groups so removing one group is unambiguous."""
    features = copy.deepcopy(config.get("data", {}).get("features") or {})
    return {
        "history": list(features.get("history", DEFAULT_FEATURE_COLUMNS.history)),
        "weather": list(features.get("weather", DEFAULT_FEATURE_COLUMNS.weather)),
        "direct": list(features.get("direct", DEFAULT_FEATURE_COLUMNS.direct)),
        "target": str(features.get("target", DEFAULT_FEATURE_COLUMNS.target)),
    }


def run_zero_feature_recursive_training(
    config: dict[str, Any],
    run_dir: Path,
    zero_group: str,
    note: str | None = None,
) -> RecursiveExperimentResult:
    """Train a recursive model after zeroing one input feature group."""
    config = recursive_forecast_config(config)
    started_at = datetime.now()
    started_timer = time.perf_counter()
    set_random_seed(int(config.get("seed", 42)))
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    write_run_note(run_dir, note)

    arrays_by_split = load_or_make_split_arrays(config)
    arrays_by_split = {name: zero_feature_group_arrays(arrays, zero_group) for name, arrays in arrays_by_split.items()}
    model_probe = build_model(config)
    if getattr(model_probe, "is_torch_model", False) and should_scale_torch_training(config):
        scaler = fit_window_scaler(arrays_by_split)
        config = attach_fitted_scaler(config, scaler)
        arrays_by_split = transform_window_arrays_by_split(arrays_by_split, scaler)
    split_sizes = {split_name: len(arrays.target) for split_name, arrays in arrays_by_split.items()}

    step_config = make_recursive_step_config(config)
    train_horizon = int(step_config["data"]["horizon"])
    step_arrays = {split_name: slice_step_arrays(arrays, 0, step_count=train_horizon) for split_name, arrays in arrays_by_split.items()}
    model = build_model(step_config)
    display_model_name = recursive_strategy_model_name(config)
    print_training_parameters({**step_config, "model": {**step_config.get("model", {}), "name": display_model_name}}, run_dir, model, started_at, split_sizes)
    fit_started_at = datetime.now()
    fit_started_timer = time.perf_counter()
    print_training_process_start(fit_started_at)
    train_result = train_model(model, step_arrays, step_config, epoch_callback=print_epoch_progress)
    fit_ended_at = datetime.now()
    print_training_process_end(fit_ended_at, time.perf_counter() - fit_started_timer, train_result.epoch_history)

    val_predictions = recursive_prediction_frame("Recursive", model, arrays_by_split["val"], config=step_config)
    predictions = recursive_prediction_frame("Recursive", model, arrays_by_split["test"], config=step_config)
    metrics = dict(train_result.metrics)
    metrics.update(recursive_prediction_metrics("val", val_predictions, config=step_config))
    metrics.update(recursive_prediction_metrics("test", predictions, config=step_config))

    model_path = save_model_artifact(model, step_config, models_dir, stem="best")
    result = RecursiveExperimentResult(
        run_dir=run_dir,
        metrics=metrics,
        predictions=predictions,
        epoch_history=train_result.epoch_history,
        model_path=model_path,
        best_epoch=train_result.best_epoch,
        duration_seconds=time.perf_counter() - started_timer,
    )
    write_recursive_outputs(run_dir, result)
    save_config(step_config, run_dir / "config.yaml")
    save_manifest(
        build_manifest(
            run_dir=run_dir,
            config={**config, "model": {**config.get("model", {}), "name": display_model_name}},
            started_at=started_at,
            ended_at=datetime.now(),
            duration_seconds=result.duration_seconds,
            split_sizes=split_sizes,
            model_path=model_path,
            best_epoch=train_result.best_epoch,
        ),
        run_dir / "manifest.json",
    )
    write_epoch_history(run_dir / "epoch_history.csv", result.epoch_history)
    write_split_metrics(run_dir / "metrics.csv", result.metrics)
    return result


def zero_feature_group_arrays(arrays: WindowArrays, group: str) -> WindowArrays:
    """Return arrays with one input group zeroed while preserving model input shapes."""
    return WindowArrays(
        history=np.zeros_like(arrays.history) if group == "history" else arrays.history,
        weather=np.zeros_like(arrays.weather) if group == "weather" else arrays.weather,
        direct=np.zeros_like(arrays.direct) if group == "direct" else arrays.direct,
        target=arrays.target,
        target_time=arrays.target_time,
    )


def ablation_row(label: str, result: RecursiveExperimentResult) -> dict[str, str]:
    row = {
        "label": label,
        "model": "recursive",
        "run_dir": str(result.run_dir),
        "duration_seconds": f"{result.duration_seconds:.3f}",
    }
    for split_name in ("test", "test_generation"):
        for metric_name in BASE_METRIC_NAMES:
            key = f"{split_name}_{metric_name}"
            if key in result.metrics:
                row[key] = str(result.metrics[key])
    return row


def write_ablation_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    fields = collect_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_fields(rows: list[dict[str, str]]) -> list[str]:
    fields = ["label", "model", "run_dir"]
    for split_name in ("test", "test_generation"):
        for metric_name in BASE_METRIC_NAMES:
            field = f"{split_name}_{metric_name}"
            if any(field in row for row in rows):
                fields.append(field)
    fields.append("duration_seconds")
    return fields


def write_ablation_summary(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["label", "test_mae", "test_rmse", "test_nmae", "test_nrmse", "duration_seconds", "run_dir"]
    lines = [
        "# Recursive 4h BNN Input Branch Ablation",
        "",
        "| Variant | Test MAE | Test RMSE | Test NMAE | Test NRMSE | Duration (s) | Run Dir |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        values = [row.get(column, "") for column in columns]
        lines.append(f"| {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {values[5]} | `{values[6]}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
