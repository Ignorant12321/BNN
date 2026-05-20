"""统一评估并对比已训练 run。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import yaml

from src.artifacts.run_io import create_comparison_dir, resolve_run_path, save_config
from src.config import load_config
from src.evaluation.evaluator import evaluate_run
from src.evaluation.plots import write_metrics_bar_svg


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="加载已训练模型并统一评估对比。")
    parser.add_argument("--config", default=None, help="对比配置文件")
    parser.add_argument("--run", action="append", default=[], help="单个 run，格式为 label=run_dir；可重复传入")
    parser.add_argument("--name", default="comparison", help="本次对比名称，仅在使用 --run 时生效")
    parser.add_argument("--output-dir", default="outputs", help="输出根目录，仅在使用 --run 时生效")
    parser.add_argument("--split", default="test", help="评估 split，默认 test")
    args = parser.parse_args()
    if args.run:
        runs = [parse_cli_run(value) for value in args.run]
        out_dir = run_compare_from_runs(runs, name=args.name, output_dir=args.output_dir, split=args.split)
    else:
        out_dir = run_compare(args.config or "configs/compare/main.yaml")
    print_compare_console_summary(out_dir / "model_metrics.txt")


def run_compare(config_path: str | Path) -> Path:
    """读取配置并执行统一评估对比。"""
    config = load_config(config_path)
    return run_compare_from_runs(
        parse_runs(config),
        name=str(config.get("name", "comparison")),
        output_dir=config.get("output_dir", "outputs"),
        split=str(config.get("split", "test")),
        compare_config=config,
    )


def run_compare_from_runs(
    runs: list[dict[str, str]],
    name: str = "comparison",
    output_dir: str | Path = "outputs",
    split: str = "test",
    compare_config: dict[str, Any] | None = None,
) -> Path:
    """加载一个或多个 run，在同一 split 上统一评估并写出对比产物。"""
    runs = parse_runs({"runs": runs})
    out_dir = create_comparison_dir(name, output_dir)
    predictions_dir = out_dir / "predictions"
    figures_dir = out_dir / "figures"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_config(compare_config or {"name": name, "output_dir": str(output_dir), "split": split, "runs": runs}, out_dir / "compare_config.yaml")

    rows = []
    for run in runs:
        run_path = resolve_run_path(run["path"])
        row = evaluate_or_read_metrics(run["label"], run_path, split, predictions_dir)
        rows.append(row)

    write_metrics_csv(out_dir / "model_metrics.csv", rows)
    write_summary(out_dir / "model_metrics.txt", rows)
    write_metrics_bar_svg(rows, figures_dir / "metrics_bar.svg", metric=f"{split}_rmse")
    write_report(out_dir / "report.md", rows, split)
    return out_dir


def evaluate_or_read_metrics(label: str, run_path: Path, split: str, predictions_dir: Path) -> dict[str, str]:
    """优先统一加载模型评估；旧 run 没有模型时退回读取已有 metrics.csv。"""
    try:
        result = evaluate_run(label, run_path, split=split)
        safe_label = safe_filename(label)
        result.predictions.to_csv(predictions_dir / f"{safe_label}.csv", index=False)
        row = {"label": label, "model": result.model_name, "run_dir": str(result.run_dir)}
        row.update({name: str(value) for name, value in result.metrics.items()})
        return row
    except (FileNotFoundError, ValueError):
        row = {"label": label, "model": run_path.parent.name, "run_dir": str(run_path)}
        row.update(read_legacy_metrics(run_path / "metrics" / "metrics.csv"))
        return row


def parse_cli_run(value: str) -> dict[str, str]:
    """解析命令行传入的单个 run。"""
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        path = value
        label = Path(path).parent.name or Path(path).name
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError("--run label must be a non-empty string")
    if not path:
        raise ValueError("--run path must be a non-empty string")
    return {"label": label, "path": path}


def parse_runs(config: dict[str, Any]) -> list[dict[str, str]]:
    """解析 compare 配置中的 runs 列表。"""
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty list")
    parsed = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be a mapping")
        label = run.get("label")
        path = run.get("path")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"runs[{index}].label must be a non-empty string")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"runs[{index}].path must be a non-empty string")
        parsed.append({"label": label.strip(), "path": path.strip()})
    return parsed


def read_legacy_metrics(path: Path) -> dict[str, str]:
    """读取旧格式 metrics.csv。"""
    if not path.is_file():
        raise FileNotFoundError(f"metrics file not found: {path}")
    metrics: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            metrics[str(row["metric"])] = str(row["value"])
    return metrics


def write_metrics_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """写出对比指标 CSV。"""
    fields = collect_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    """写出对比汇总 TXT。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_summary_table(rows), encoding="utf-8")


def format_summary_table(rows: list[dict[str, str]]) -> str:
    """把对比结果格式化成适合终端和 txt 文件阅读的表格。"""
    fields = collect_fields(rows)
    widths = {field: max(len(field), *(len(str(row.get(field, ""))) for row in rows)) for field in fields}
    header_line = " | ".join(field.ljust(widths[field]) for field in fields)
    separator = "-+-".join("-" * widths[field] for field in fields)
    body = [" | ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields) for row in rows]
    return "\n".join(["PV Forecast Result Comparison", "", header_line, separator, *body, ""]) + "\n"


def collect_fields(rows: list[dict[str, str]]) -> list[str]:
    """收集表格字段，保持主要字段靠前。"""
    fields = ["label", "model", "run_dir"]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_report(path: Path, rows: list[dict[str, str]], split: str) -> None:
    """写出 Markdown 简报。"""
    lines = ["# PV Forecast Comparison", "", f"Split: `{split}`", "", format_summary_table(rows)]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_compare_console_summary(summary_path: Path) -> None:
    """在控制台打印对比摘要。"""
    print(colorize("Compare Summary", "cyan"))
    print(summary_path.read_text(encoding="utf-8"))
    print(f"{colorize('Saved:', 'green')} {summary_path}")


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
