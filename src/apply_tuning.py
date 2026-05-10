"""Apply Optuna best parameters to the default training config.

The command intentionally migrates only the parameters searched in
``src.tune``. It does not copy ``best_config.yaml`` wholesale, so unrelated
changes in tuning configs cannot leak into the formal training config.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import load_config, save_config


PARAM_PATHS = {
    "hidden_dim": ("model", "hidden_dim"),
    "branch_dim": ("model", "branch_dim"),
    "lr": ("training", "lr"),
    "kl_beta": ("training", "kl_beta"),
}


@dataclass(frozen=True)
class ParamChange:
    """A single attempted parameter migration."""

    path: str
    old: Any
    new: Any

    @property
    def status(self) -> str:
        return "changed" if self.old != self.new else "unchanged"

    def as_row(self) -> dict[str, Any]:
        return {"path": self.path, "old": self.old, "new": self.new, "status": self.status}


def load_best_params(path: str | Path) -> dict[str, Any]:
    """Read the ``best_params`` object from an Optuna ``best_params.json`` file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        raise ValueError(f"{path} does not contain a best_params object")
    return best_params


def load_summary_objective(path: str | Path) -> str:
    """Read the objective metric recorded in a tuning summary.

    Older summaries did not store this field; those runs optimized RMSE.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return str(payload.get("objective_metric", "rmse"))


def resolve_default_objective(config_path: str | Path = "configs/tuning.yaml") -> str:
    """Read the default apply-tuning objective from the tuning config."""
    config = load_config(config_path)
    return str(config.get("tuning", {}).get("objective_metric", "rmse"))


def find_latest_best_params(tuning_dir: str | Path = "outputs/tuning", objective_metric: str | None = None) -> Path:
    """Return the newest timestamped tuning summary, optionally filtered by objective."""
    tuning_path = Path(tuning_dir)
    candidates = sorted(tuning_path.glob("*/best_params.json"), key=lambda path: path.parent.name)
    if objective_metric is not None:
        candidates = [path for path in candidates if load_summary_objective(path) == objective_metric]
    if not candidates:
        suffix = f" for objective_metric={objective_metric!r}" if objective_metric is not None else ""
        raise FileNotFoundError(f"No best_params.json found under {tuning_path}{suffix}")
    return candidates[-1]


def apply_best_params(source: str | Path, target: str | Path, dry_run: bool = False) -> list[ParamChange]:
    """Apply supported best parameters to a YAML config and report every touched value."""
    best_params = load_best_params(source)
    config = load_config(target)
    changes: list[ParamChange] = []
    has_write = False

    for param_name, path_parts in PARAM_PATHS.items():
        if param_name not in best_params:
            continue

        section_name, value_name = path_parts
        section = config.setdefault(section_name, {})
        old_value = section.get(value_name)
        new_value = best_params[param_name]
        change = ParamChange(".".join(path_parts), old_value, new_value)
        changes.append(change)
        if change.status == "changed":
            section[value_name] = new_value
            has_write = True

    if has_write and not dry_run:
        save_config(config, target)

    return changes


def format_change(change: ParamChange) -> str:
    """Format one change line for CLI output."""
    return f"{change.path}: {change.old} -> {change.new} ({change.status})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate Optuna best_params.json values into configs/default.yaml.")
    parser.add_argument(
        "--source",
        default="latest",
        help="Path to best_params.json, or 'latest' to use newest outputs/tuning/*/best_params.json.",
    )
    parser.add_argument(
        "--objective",
        default=None,
        help="Objective metric to select when --source latest. Defaults to tuning.objective_metric from --config.",
    )
    parser.add_argument("--config", default="configs/tuning.yaml", help="Tuning config used to resolve the default objective.")
    parser.add_argument("--target", default="configs/default.yaml", help="YAML config to update.")
    parser.add_argument("--tuning-dir", default="outputs/tuning", help="Directory used when --source latest.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing the target config.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    objective = None
    if args.source == "latest":
        objective = args.objective or resolve_default_objective(args.config)
        source = find_latest_best_params(args.tuning_dir, objective_metric=objective)
    else:
        source = Path(args.source)
    changes = apply_best_params(source, args.target, dry_run=args.dry_run)

    print(f"Source: {source}")
    print(f"Target: {args.target}")
    if args.source == "latest":
        print(f"Objective: {objective}")
    if args.dry_run:
        print("Mode: dry-run")
    if not changes:
        print("No supported parameters found in best_params.json.")
        return
    for change in changes:
        print(format_change(change))


if __name__ == "__main__":
    main()
