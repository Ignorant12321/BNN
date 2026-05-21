"""单模型训练入口。

功能：
    读取配置和已切分的 train/val/test CSV，在 train split 上拟合模型，
    评估 train/val，并输出训练历史和图表。

使用：
    python -m src.experiments.train --config configs/models/bnn/24h.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.artifacts.manifest import build_manifest
from src.artifacts.run_io import create_run_dir, save_config, save_manifest, save_model_artifact
from src.config import load_config
from src.data.pv import load_split_window_arrays_from_config
from src.data.scaling import attach_fitted_scaler, fit_window_scaler, should_scale_torch_training, transform_window_arrays_by_split
from src.evaluation.metrics import evaluate_arrays
from src.evaluation.plots import write_training_loss_png
from src.models.registry import build_model
from src.torch_runtime import import_torch
from src.training.trainer import train_model


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Train one PV forecasting model.")
    parser.add_argument("--config", default="configs/models/bnn/24h.yaml")
    args = parser.parse_args()
    run_training(load_config(args.config))


def run_training(config: dict[str, Any]) -> Path:
    """执行一次单模型训练：只用 train split 拟合，并评估 train/val。"""
    started_at = datetime.now()
    started_timer = time.perf_counter()
    set_random_seed(int(config.get("seed", 42)))
    model = build_model(config)
    run_dir = create_run_dir(config)
    models_dir = run_dir / "models"
    figures_dir = run_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    arrays_by_split = load_or_make_split_arrays(config)
    if getattr(model, "is_torch_model", False) and should_scale_torch_training(config):
        scaler = fit_window_scaler(arrays_by_split)
        config = attach_fitted_scaler(config, scaler)
        arrays_by_split = transform_window_arrays_by_split(arrays_by_split, scaler)
    split_sizes = {split_name: len(arrays.target) for split_name, arrays in arrays_by_split.items()}
    print_training_parameters(config, run_dir, model, started_at, split_sizes)
    fit_started_at = datetime.now()
    fit_started_timer = time.perf_counter()
    print_training_process_start(fit_started_at)
    train_result = train_model(model, arrays_by_split, config, epoch_callback=print_epoch_progress)
    fit_ended_at = datetime.now()
    fit_duration_seconds = time.perf_counter() - fit_started_timer
    print_training_process_end(fit_ended_at, fit_duration_seconds, train_result.epoch_history)

    save_config(config, run_dir / "config.yaml")
    write_epoch_history(run_dir / "epoch_history.csv", train_result.epoch_history)
    write_split_metrics(run_dir / "metrics.csv", train_result.metrics)
    write_training_loss_png(train_result.epoch_history, train_result.metrics, figures_dir / "loss_curve.png")
    model_path = save_model_artifact(model, config, models_dir, stem="best")
    ended_at = datetime.now()
    duration_seconds = time.perf_counter() - started_timer
    log_path = run_dir / "train.log"
    write_training_log(
        log_path=log_path,
        config=config,
        run_dir=run_dir,
        model=model,
        started_at=started_at,
        fit_started_at=fit_started_at,
        fit_ended_at=fit_ended_at,
        fit_duration_seconds=fit_duration_seconds,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        split_sizes=split_sizes,
        metrics=train_result.metrics,
        epoch_history=train_result.epoch_history,
        model_path=model_path,
    )
    save_manifest(
        build_manifest(
            run_dir=run_dir,
            config=config,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            split_sizes=split_sizes,
            model_path=model_path,
            best_epoch=train_result.best_epoch,
        ),
        run_dir / "manifest.json",
    )
    print_training_results(
        run_dir=run_dir,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        metrics=train_result.metrics,
        log_path=log_path,
        model_path=model_path,
    )
    return run_dir


def evaluate_model(model, arrays) -> dict[str, float]:
    """在传入 split 的全部窗口上评估拟合后的模型。"""
    return evaluate_arrays(model, arrays)


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = import_torch()
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_make_split_arrays(config: dict[str, Any]) -> dict[str, Any]:
    """优先读取已切分 CSV；真实数据存在但未切分时拒绝全量训练以避免泄露。"""
    data = config.get("data", {})
    processed_dir = Path(data.get("processed_dir", "data/processed"))
    split_paths = {split_name: processed_dir / f"{split_name}.csv" for split_name in ("train", "val", "test")}
    existing_splits = [path for path in split_paths.values() if path.is_file()]
    if len(existing_splits) == len(split_paths):
        return load_split_window_arrays_from_config(config)

    generation_path = Path(data.get("generation_path", ""))
    weather_path = Path(data.get("weather_path", ""))
    if generation_path.is_file() and weather_path.is_file():
        missing = ", ".join(str(path) for path in split_paths.values() if not path.is_file())
        raise FileNotFoundError(
            "missing split CSV files; run `python -m src.data.preprocess --config configs/data.yaml` "
            f"and `python -m src.data.split --config configs/data.yaml` first. Missing: {missing}"
        )
    smoke = make_smoke_arrays(config)
    return {"train": smoke, "val": smoke, "test": smoke}


def write_epoch_history(path: Path, epoch_history: list[dict[str, float]]) -> None:
    """写出每轮训练 loss。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "loss"]
    if any("val_rmse" in item for item in epoch_history):
        fieldnames.append("val_rmse")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in epoch_history:
            row = {"epoch": int(item["epoch"]), "loss": item["loss"]}
            if "val_rmse" in item:
                row["val_rmse"] = item["val_rmse"]
            writer.writerow(row)


def write_split_metrics(path: Path, metrics: dict[str, float]) -> None:
    """写出 train/val 分 split 指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        for name, value in metrics.items():
            if "_" not in name:
                continue
            split_name, metric_name = name.split("_", 1)
            writer.writerow({"split": split_name, "metric": metric_name, "value": value})


def write_training_log(
    log_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    model,
    started_at: datetime,
    fit_started_at: datetime,
    fit_ended_at: datetime,
    fit_duration_seconds: float,
    ended_at: datetime,
    duration_seconds: float,
    split_sizes: dict[str, int],
    metrics: dict[str, float],
    epoch_history: list[dict[str, float]],
    model_path: Path,
) -> None:
    """写出训练日志文件。"""
    process_rows = [
        ("Fit Start", format_time(fit_started_at)),
        *[(f"Epoch {int(item['epoch'])}", f"loss={item['loss']:.6f}") for item in epoch_history],
        ("Fit End", format_time(fit_ended_at)),
        ("Fit Duration", format_duration(fit_duration_seconds)),
    ]
    if not epoch_history:
        process_rows.append(("Fit", "completed"))
    lines = [
        *format_plain_section(
            "Training Parameters",
            training_parameter_rows(config, run_dir, model, started_at, split_sizes),
        ),
        "",
        *format_plain_section("Training Process", process_rows),
        "",
        *format_plain_section(
            "Training Results",
            training_result_rows(run_dir, ended_at, duration_seconds, metrics, log_path, model_path),
        ),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_training_parameters(
    config: dict[str, Any],
    run_dir: Path,
    model,
    started_at: datetime,
    split_sizes: dict[str, int],
) -> None:
    """在控制台打印训练参数区。"""
    print_section("Training Parameters", training_parameter_rows(config, run_dir, model, started_at, split_sizes))


def training_parameter_rows(
    config: dict[str, Any],
    run_dir: Path,
    model,
    started_at: datetime,
    split_sizes: dict[str, int],
) -> list[tuple[str, str]]:
    """返回训练参数区行。"""
    training = config.get("training", {})
    data = config.get("data", {})
    return [
        ("Model", str(config.get("model", {}).get("name", "model"))),
        ("Backend", str(training.get("backend", "numpy"))),
        ("Device", str(training.get("device", "auto"))),
        ("Lookback/Horizon", f"{data.get('lookback', '')}/{data.get('horizon', '')}"),
        ("Windows", ", ".join(f"{name}={count}" for name, count in split_sizes.items())),
        ("Epochs", str(training.get("epochs", ""))),
        ("Batch Size", str(training.get("batch_size", ""))),
        ("Learning Rate", str(training.get("lr", ""))),
        ("Weight Decay", str(training.get("weight_decay", ""))),
        ("Parameters", str(model_parameter_count(model))),
        ("Start Time", format_time(started_at)),
        ("Output", str(run_dir)),
    ]


def print_training_process_start(fit_started_at: datetime) -> None:
    """在控制台打印训练过程开始信息。"""
    print_section("Training Process", [("Fit Start", format_time(fit_started_at))])


def print_epoch_progress(item: dict[str, float]) -> None:
    """每个 epoch 结束后立即打印一行 loss。"""
    print_rows([(f"Epoch {int(item['epoch'])}", f"loss={item['loss']:.6f}")])


def print_training_process_end(
    fit_ended_at: datetime,
    fit_duration_seconds: float,
    epoch_history: list[dict[str, float]],
) -> None:
    """在控制台打印训练过程结束信息。"""
    rows = [
        ("Fit End", format_time(fit_ended_at)),
        ("Fit Duration", format_duration(fit_duration_seconds)),
    ]
    if not epoch_history:
        rows.append(("Fit", "completed"))
    print_rows(rows)


def print_training_results(
    run_dir: Path,
    ended_at: datetime,
    duration_seconds: float,
    metrics: dict[str, float],
    log_path: Path,
    model_path: Path,
) -> None:
    """在控制台打印训练结果区。"""
    print_section("Training Results", training_result_rows(run_dir, ended_at, duration_seconds, metrics, log_path, model_path))
    print(colorize("Result files written.", "green"), flush=True)


def training_result_rows(
    run_dir: Path,
    ended_at: datetime,
    duration_seconds: float,
    metrics: dict[str, float],
    log_path: Path,
    model_path: Path,
) -> list[tuple[str, str]]:
    """返回训练结果区行。"""
    return [
        ("End Time", format_time(ended_at)),
        ("Duration", format_duration(duration_seconds)),
        ("Val MAE", format_metric(metrics.get("val_mae"))),
        ("Val RMSE", format_metric(metrics.get("val_rmse"))),
        ("Val NMAE", format_metric(metrics.get("val_nmae"))),
        ("Val NRMSE", format_metric(metrics.get("val_nrmse"))),
        ("Output", str(run_dir)),
        ("Log", str(log_path)),
        ("Model File", str(model_path)),
    ]


def print_section(title: str, rows: list[tuple[str, str]]) -> None:
    """打印带标题的键值区块。"""
    print(flush=True)
    print(colorize(title, "cyan"), flush=True)
    print("-" * max(48, max(len(name) for name, _ in rows) + 34), flush=True)
    print_rows(rows)


def print_rows(rows: list[tuple[str, str]]) -> None:
    """打印对齐的键值行。"""
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        label = colorize(name.ljust(width), "yellow")
        print(f"{label} : {value}", flush=True)


def format_plain_section(title: str, rows: list[tuple[str, str]]) -> list[str]:
    """返回和控制台同结构的纯文本区块。"""
    width = max(len(name) for name, _ in rows)
    return [
        title,
        "-" * max(48, max(len(name) for name, _ in rows) + 34),
        *[f"{name.ljust(width)} : {value}" for name, value in rows],
    ]


def model_parameter_count(model) -> int:
    """返回模型参数量。"""
    if hasattr(model, "parameters"):
        return int(sum(parameter.numel() for parameter in model.parameters()))
    coef = getattr(model, "coef_", None)
    if coef is not None:
        return int(np.size(coef))
    return 0


def format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 1:
        return f"{remaining:.2f} s"
    return f"{int(minutes)} min {remaining:.2f} s"


def format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def colorize(text: str, color: str) -> str:
    """返回适度着色的终端文本；设置 NO_COLOR 时禁用颜色。"""
    if os.environ.get("NO_COLOR"):
        return text
    codes = {"cyan": "36", "green": "32", "yellow": "33"}
    code = codes.get(color)
    if code is None:
        return text
    return f"\033[{code}m{text}\033[0m"


def make_smoke_arrays(config: dict[str, Any]):
    """在无真实数据的测试/冒烟场景下构造确定性零数组。"""
    from src.data.pv import WindowArrays, feature_dimensions_from_config

    data = config["data"]
    history_features, weather_features, direct_features = feature_dimensions_from_config(config)
    batch_size = int(config.get("training", {}).get("batch_size", 4))
    horizon = int(data["horizon"])
    return WindowArrays(
        history=np.zeros((batch_size, int(data["lookback"]), history_features), dtype=np.float32),
        weather=np.zeros((batch_size, horizon, weather_features), dtype=np.float32),
        direct=np.zeros((batch_size, direct_features), dtype=np.float32),
        target=np.zeros((batch_size, horizon), dtype=np.float32),
        target_time=None,
    )


if __name__ == "__main__":
    main()
