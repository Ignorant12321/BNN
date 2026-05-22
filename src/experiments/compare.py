"""统一评估并对比已训练产物。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from src.artifacts.run_io import create_comparison_dir, is_run_dir, resolve_run_path, save_config, write_run_note
from src.config import load_config
from src.evaluation.metrics import BASE_METRIC_NAMES
from src.evaluation.evaluator import evaluate_run
from src.evaluation.plots import write_comparison_loss_png, write_prediction_window_metrics_csv, write_prediction_window_pngs


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="加载已训练模型并统一评估对比。")
    parser.add_argument("--config", default=None, help="对比配置文件")
    parser.add_argument("--run", action="append", default=[], help="训练产物或模型根目录；支持 path 或 label=path，可重复传入")
    parser.add_argument("--name", default="comparison", help="本次对比名称，仅在使用 --run 时生效")
    parser.add_argument("--output-dir", default="outputs", help="输出根目录，仅在使用 --run 时生效")
    parser.add_argument("--split", default="test", help="评估 split，默认 test")
    parser.add_argument("--note", default=None, help="写入对比输出目录 note.txt 的备注；默认写入时间戳目录名")
    args = parser.parse_args()
    if args.run:
        runs = [parse_cli_run(value) for value in args.run]
        out_dir = run_compare_from_runs(runs, name=args.name, output_dir=args.output_dir, split=args.split, note=args.note)
    else:
        out_dir = run_compare(args.config or "configs/compare/main.yaml", note=args.note)
    print_compare_console_summary(out_dir / "model_metrics.csv")


def run_compare(config_path: str | Path, note: str | None = None) -> Path:
    """读取配置并执行统一评估对比。"""
    config = load_config(config_path)
    return run_compare_from_runs(
        parse_runs(config),
        name=str(config.get("name", "comparison")),
        output_dir=config.get("output_dir", "outputs"),
        split=str(config.get("split", "test")),
        compare_config=config,
        note=note,
    )


def run_compare_from_runs(
    runs: list[dict[str, str]],
    name: str = "comparison",
    output_dir: str | Path = "outputs",
    split: str = "test",
    compare_config: dict[str, Any] | None = None,
    note: str | None = None,
) -> Path:
    """加载一个或多个训练产物，在同一 split 上统一评估并写出对比产物。"""
    runs = parse_runs({"runs": runs})
    out_dir = create_comparison_dir(name, output_dir)
    predictions_dir = out_dir / "predictions"
    figures_dir = out_dir / "figures"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_run_note(out_dir, note)
    save_config(compare_config or {"name": name, "output_dir": str(output_dir), "split": split, "runs": runs}, out_dir / "compare_config.yaml")

    rows = []
    prediction_frames = []
    loss_histories = []
    for run in runs:
        run_path = resolve_run_path(run["path"])
        result = evaluate_run(run["label"], run_path, split=split)
        safe_label = safe_filename(run["label"])
        result.predictions.to_csv(predictions_dir / f"{safe_label}.csv", index=False)
        row = {"label": run["label"], "model": result.model_name, "run_dir": str(result.run_dir)}
        row.update({name: str(value) for name, value in result.metrics.items()})
        rows.append(row)
        prediction_frames.append(result.predictions)
        loss_histories.append({"label": run["label"], "history": read_epoch_history(run_path)})

    write_metrics_csv(out_dir / "model_metrics.csv", rows)
    write_prediction_window_pngs(prediction_frames, figures_dir)
    if prediction_frames:
        import pandas as pd

        write_prediction_window_metrics_csv(pd.concat(prediction_frames, ignore_index=True), figures_dir / "prediction_window_metrics.csv")
    write_comparison_loss_png(loss_histories, figures_dir / "loss_curves.png")
    return out_dir


def parse_cli_run(value: str) -> dict[str, str]:
    """解析命令行传入的单个训练产物或模型根目录。"""
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
    else:
        path = value
        label = ""
    path = path.strip()
    if not path:
        raise ValueError("--run path must be a non-empty string")
    run = {"path": path}
    if label:
        run["label"] = label
    return run


def parse_runs(config: dict[str, Any]) -> list[dict[str, str]]:
    """解析 compare 配置中的 runs 列表，允许省略 label。"""
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty list")
    parsed = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be a mapping")
        label = run.get("label")
        path = run.get("path")
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError(f"runs[{index}].label must be a non-empty string when provided")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"runs[{index}].path must be a non-empty string")
        parsed_run = {"path": path.strip()}
        if isinstance(label, str):
            parsed_run["label"] = label.strip()
        parsed.append(parsed_run)
    return expand_run_entries(parsed)


def expand_run_entries(runs: list[dict[str, str]]) -> list[dict[str, str]]:
    """把模型根目录展开为其中全部训练 run；单个 run 默认用目录名做 label。"""
    expanded: list[dict[str, str]] = []
    for run in runs:
        path = Path(run["path"])
        child_runs = child_run_dirs(path)
        if child_runs:
            for child in child_runs:
                expanded.append({"label": child.name, "path": str(child)})
            continue
        expanded.append({"label": run.get("label") or path.name, "path": str(path)})
    return expanded


def child_run_dirs(path: Path) -> list[Path]:
    """返回 path 下的直接训练产物子目录；path 本身是 run 时不展开。"""
    if is_run_dir(path) or not path.is_dir():
        return []
    return sorted((child for child in path.iterdir() if child.is_dir() and is_run_dir(child)), key=lambda child: child.name)


def write_metrics_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """写出对比指标 CSV。"""
    fields = collect_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_summary_table(rows: list[dict[str, str]]) -> str:
    """把对比结果格式化成适合终端阅读的竖排摘要。"""
    fields = collect_fields(rows)
    field_width = max(len(field) for field in fields)
    lines = ["PV Forecast Result Comparison", ""]
    for index, row in enumerate(rows, start=1):
        label = row.get("label", f"run-{index}")
        lines.append(f"Run {index}: {label}")
        lines.append("-" * (len(lines[-1])))
        for field in fields:
            lines.append(f"{field.ljust(field_width)} : {row.get(field, '')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def collect_fields(rows: list[dict[str, str]]) -> list[str]:
    """收集表格字段，保持主要字段靠前。"""
    fields = ["label", "model", "run_dir"]
    for split_name in ("train", "val", "test", "test_generation"):
        for metric_name in BASE_METRIC_NAMES:
            field = f"{split_name}_{metric_name}"
            if any(field in row for row in rows):
                fields.append(field)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def print_compare_console_summary(summary_path: Path) -> None:
    """在控制台打印对比摘要。"""
    print(colorize("Compare Summary", "cyan"))
    rows = read_metrics_rows(summary_path)
    print(format_summary_table(rows))
    print(f"{colorize('Saved:', 'green')} {summary_path}")


def read_metrics_rows(path: Path) -> list[dict[str, str]]:
    """读取 compare 输出的 model_metrics.csv。"""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_epoch_history(run_path: Path) -> list[dict[str, float]]:
    """读取单个训练产物的 epoch loss。"""
    history_path = run_path / "epoch_history.csv"
    if not history_path.is_file():
        history_path = run_path / "metrics" / "epoch_history.csv"
    if not history_path.is_file():
        return []
    history = []
    with history_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("epoch") and row.get("loss"):
                history.append({"epoch": float(row["epoch"]), "loss": float(row["loss"])})
    return history


def metric_names_for_split(split: str) -> list[str]:
    """返回 compare 图表使用的指标名。"""
    return [f"{split}_{name}" for name in BASE_METRIC_NAMES]


def safe_filename(value: str) -> str:
    """把 label 转成适合文件名的字符串。"""
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "run"


def colorize(text: str, color: str) -> str:
    """返回适度着色的终端文本；设置 NO_COLOR 时禁用颜色。"""
    if os.environ.get("NO_COLOR"):
        return text
    codes = {"cyan": "36", "green": "32", "yellow": "33"}
    code = codes.get(color)
    if code is None:
        return text
    return f"\033[{code}m{text}\033[0m"


if __name__ == "__main__":
    main()
