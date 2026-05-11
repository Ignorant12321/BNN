"""Select existing tuning trials by validation metrics.

This command reuses completed Optuna trial outputs. It does not launch new
training; it reads each trial's ``validation_metrics.json`` and can update a
tuning session's ``best_params.json`` and ``best_config.yaml`` after an explicit
confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.apply_tuning import PARAM_PATHS
from src.utils import load_config, save_config


DEFAULT_SHOW_METRICS = ("crps", "rmse", "mae", "nll", "smape", "nrmse")


@dataclass(frozen=True)
class TrialSelection:
    """A completed tuning trial selected by one validation metric."""

    number: int
    run_dir: Path
    metric: str
    metric_value: float
    params: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True)
class ArtifactChange:
    """A single change planned for a tuning summary artifact."""

    path: str
    old: Any
    new: Any

    @property
    def status(self) -> str:
        return "changed" if self.old != self.new else "unchanged"


def find_latest_tuning_run(tuning_dir: str | Path = "outputs/tuning") -> Path:
    """Return the newest tuning session directory that contains ``trials.csv``."""
    root = Path(tuning_dir)
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and (path / "trials.csv").exists())
    if not candidates:
        raise FileNotFoundError(f"No tuning run with trials.csv found under {root}")
    return candidates[-1]


def resolve_source(source: str | Path, tuning_dir: str | Path = "outputs/tuning") -> Path:
    """Resolve ``latest`` or a concrete tuning session directory."""
    if str(source) == "latest":
        return find_latest_tuning_run(tuning_dir)
    path = Path(source)
    if not (path / "trials.csv").exists():
        raise FileNotFoundError(f"{path} does not contain trials.csv")
    return path


def load_trial_selections(tuning_run: str | Path) -> list[TrialSelection]:
    """Load completed trials and their validation metrics from a tuning session."""
    run_path = Path(tuning_run)
    with open(run_path / "trials.csv", newline="", encoding="utf-8") as f:
        trial_rows = [row for row in csv.DictReader(f) if row.get("state", "COMPLETE") == "COMPLETE"]

    metric_paths = sorted(run_path.glob("*/20*/metrics/validation_metrics.json"))
    if len(metric_paths) < len(trial_rows):
        raise ValueError(
            f"Found {len(trial_rows)} complete trials in {run_path / 'trials.csv'}, "
            f"but only {len(metric_paths)} validation metric files"
        )

    selections: list[TrialSelection] = []
    for row, metrics_path in zip(trial_rows, metric_paths, strict=False):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = {key: float(value) for key, value in json.load(f).items()}
        params = _extract_params(row)
        selections.append(
            TrialSelection(
                number=int(row["number"]),
                run_dir=metrics_path.parents[1],
                metric="",
                metric_value=float("nan"),
                params=params,
                metrics=metrics,
            )
        )
    return selections


def select_best_trial(tuning_run: str | Path, metric: str) -> TrialSelection:
    """Select the completed trial with the smallest validation metric."""
    selections = load_trial_selections(tuning_run)
    matching = [selection for selection in selections if metric in selection.metrics]
    if not matching:
        available = sorted({key for selection in selections for key in selection.metrics})
        raise KeyError(f"Metric {metric!r} not found. Available metrics: {', '.join(available)}")
    best = min(matching, key=lambda selection: selection.metrics[metric])
    return TrialSelection(
        number=best.number,
        run_dir=best.run_dir,
        metric=metric,
        metric_value=best.metrics[metric],
        params=best.params,
        metrics=best.metrics,
    )


def build_artifact_changes(tuning_run: str | Path, selection: TrialSelection) -> list[ArtifactChange]:
    """Compare selected trial data with current tuning summary artifacts."""
    run_path = Path(tuning_run)
    summary = _load_best_params_summary(run_path)
    best_config = _load_best_config(run_path)
    changes = [
        ArtifactChange("best_params.json objective_metric", summary.get("objective_metric"), selection.metric),
        ArtifactChange("best_params.json best_trial", summary.get("best_trial"), selection.number),
        ArtifactChange("best_params.json best_value", summary.get("best_value"), selection.metric_value),
    ]
    for name in PARAM_PATHS:
        if name in selection.params:
            changes.append(
                ArtifactChange(f"best_params.json best_params.{name}", summary.get("best_params", {}).get(name), selection.params[name])
            )
    for name, path_parts in PARAM_PATHS.items():
        if name not in selection.params:
            continue
        section_name, value_name = path_parts
        changes.append(
            ArtifactChange(
                f"best_config.yaml {section_name}.{value_name}",
                best_config.get(section_name, {}).get(value_name),
                selection.params[name],
            )
        )
    changes.append(
        ArtifactChange(
            "best_config.yaml tuning.objective_metric",
            best_config.get("tuning", {}).get("objective_metric"),
            selection.metric,
        )
    )
    return changes


def update_tuning_artifacts(tuning_run: str | Path, selection: TrialSelection) -> list[ArtifactChange]:
    """Write the selected trial into ``best_params.json`` and ``best_config.yaml``."""
    run_path = Path(tuning_run)
    changes = build_artifact_changes(run_path, selection)

    summary = _load_best_params_summary(run_path)
    summary["objective_metric"] = selection.metric
    summary["best_trial"] = selection.number
    summary["best_value"] = float(selection.metric_value)
    summary["best_params"] = dict(selection.params)
    with open(run_path / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    best_config = _load_best_config(run_path)
    for name, path_parts in PARAM_PATHS.items():
        if name not in selection.params:
            continue
        section_name, value_name = path_parts
        best_config.setdefault(section_name, {})[value_name] = selection.params[name]
    best_config.setdefault("tuning", {})["objective_metric"] = selection.metric
    save_config(best_config, run_path / "best_config.yaml")

    return changes


def format_selection(selection: TrialSelection, changes: list[ArtifactChange] | None = None) -> str:
    """Format one selected trial and optional artifact changes for console output."""
    lines = [
        f"Selected trial: {selection.number}",
        f"metric: {selection.metric}={selection.metric_value:.6f}",
        f"run_dir: {selection.run_dir}",
        "Validation metrics:",
    ]
    for key in DEFAULT_SHOW_METRICS:
        if key in selection.metrics:
            lines.append(f"  {key}: {selection.metrics[key]:.6f}")
    for key in sorted(set(selection.metrics) - set(DEFAULT_SHOW_METRICS)):
        lines.append(f"  {key}: {selection.metrics[key]:.6f}")
    lines.append("Params:")
    for key in PARAM_PATHS:
        if key in selection.params:
            lines.append(f"  {key}: {selection.params[key]}")
    if changes is not None:
        lines.append("Artifact changes:")
        lines.extend(f"  {change.path}: {change.old} -> {change.new} ({change.status})" for change in changes)
    return "\n".join(lines)


def format_show_summary(tuning_run: str | Path, metrics: tuple[str, ...] = DEFAULT_SHOW_METRICS) -> str:
    """Format the best trial for each requested metric."""
    lines = [f"Source: {Path(tuning_run)}", "Best trials by validation metric:"]
    for metric in metrics:
        try:
            selected = select_best_trial(tuning_run, metric)
        except KeyError:
            continue
        lines.append(
            f"  {metric}: trial={selected.number} value={selected.metric_value:.6f} "
            f"rmse={selected.metrics.get('rmse', float('nan')):.6f} "
            f"mae={selected.metrics.get('mae', float('nan')):.6f} "
            f"crps={selected.metrics.get('crps', float('nan')):.6f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query existing tuning trials and optionally update tuning summary artifacts.")
    parser.add_argument("--source", default="latest", help="Tuning session directory, or 'latest'.")
    parser.add_argument("--tuning-dir", default="outputs/tuning", help="Root directory used when --source latest.")
    parser.add_argument("--metric", help="Validation metric used to select one trial, such as rmse, mae, nll, or crps.")
    parser.add_argument("--show", action="store_true", help="Show the best trial for common validation metrics.")
    parser.add_argument("--apply", action="store_true", help="Ask for confirmation, then update best_params.json and best_config.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Show selected trial and changes without writing.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    tuning_run = resolve_source(args.source, args.tuning_dir)

    if args.show:
        print(format_show_summary(tuning_run))

    if not args.metric:
        if args.show:
            return
        parser.error("--metric is required unless --show is used")

    selected = select_best_trial(tuning_run, args.metric)
    changes = build_artifact_changes(tuning_run, selected)
    print(format_selection(selected, changes))

    if args.dry_run:
        print("Mode: dry-run")
        return
    if not args.apply:
        print("Preview only. Add --apply to update the target config.")
        return

    try:
        answer = input("Apply these changes? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in {"y", "yes"}:
        print("No changes written.")
        return

    update_tuning_artifacts(tuning_run, selected)
    print(f"Updated tuning artifacts in {tuning_run}")


def _extract_params(row: dict[str, str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for param_name in PARAM_PATHS:
        key = f"params_{param_name}"
        if key not in row or row[key] == "":
            continue
        params[param_name] = _parse_param_value(param_name, row[key])
    return params


def _load_best_params_summary(tuning_run: Path) -> dict[str, Any]:
    path = tuning_run / "best_params.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_best_config(tuning_run: Path) -> dict[str, Any]:
    path = tuning_run / "best_config.yaml"
    if not path.exists():
        return {}
    return load_config(path)


def _parse_param_value(name: str, value: str) -> Any:
    if name in {"hidden_dim", "branch_dim"}:
        return int(value)
    if name in {"lr", "kl_beta"}:
        return float(value)
    return value


if __name__ == "__main__":
    main()
