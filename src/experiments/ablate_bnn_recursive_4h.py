"""Run Recursive 4h BNN input-branch ablations.

Usage:
    python -m src.experiments.ablate_bnn_recursive_4h
    python -m src.experiments.ablate_bnn_recursive_4h --epochs 50 --n-samples 30
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import load_config
from src.data.pv import DEFAULT_FEATURE_COLUMNS
from src.evaluation.metrics import BASE_METRIC_NAMES
from src.experiments.train_bnn_recursive_4h import (
    DEFAULT_RECURSIVE_BNN_4H_CONFIG,
    RecursiveExperimentResult,
    recursive_runtime_config,
    run_recursive_training,
)


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
        result = run_recursive_training(config, run_dir=run_dir, note=f"Recursive 4h BNN ablation: {spec.label}")
        rows.append(ablation_row(spec.label, result))

    write_ablation_metrics(compare_dir / "model_metrics.csv", rows)
    write_ablation_summary(compare_dir / "summary.md", rows)
    return compare_dir


def ablation_config(config: dict[str, Any], spec: AblationSpec) -> dict[str, Any]:
    """Return a Recursive 4h config with one input feature group removed."""
    result = copy.deepcopy(config)
    result.setdefault("data", {})
    result["data"]["features"] = feature_groups(result)
    if spec.remove_feature_group is not None:
        result["data"]["features"][spec.remove_feature_group] = []
    result.setdefault("strategy", {})
    result["strategy"]["ablation"] = spec.label
    return result


def feature_groups(config: dict[str, Any]) -> dict[str, Any]:
    """Return explicit feature groups so removing one group is unambiguous."""
    features = copy.deepcopy(config.get("data", {}).get("features") or {})
    return {
        "history": list(features.get("history", DEFAULT_FEATURE_COLUMNS.history)),
        "weather": list(features.get("weather", DEFAULT_FEATURE_COLUMNS.weather)),
        "direct": list(features.get("direct", DEFAULT_FEATURE_COLUMNS.direct)),
        "target": str(features.get("target", DEFAULT_FEATURE_COLUMNS.target)),
    }


def ablation_row(label: str, result: RecursiveExperimentResult) -> dict[str, str]:
    row = {
        "label": label,
        "model": "improved_bnn_recursive",
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
