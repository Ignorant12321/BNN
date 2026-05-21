"""把 Optuna 最优参数应用到模型 YAML。

功能：
    读取 tuning 目录中的 `tuning_config.yaml` 和 `best_config.yaml`，
    展示 search_space 覆盖到的参数变化，并在确认后写回目标模型配置。

使用：
    python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/bnn_optuna --target configs/models/bnn/bnn.yaml
    python -m src.experiments.apply_tuning --tuning-dir outputs/tuning/bnn_optuna --target configs/models/bnn/bnn.yaml --yes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import yaml

from src.config import load_config
from src.experiments.tune import normalize_param_path, set_config_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Optuna best parameters to a model YAML.")
    parser.add_argument("--tuning-dir", default="outputs/tuning/bnn_optuna", help="调参输出目录")
    parser.add_argument("--target", default="configs/models/bnn/bnn.yaml", help="要更新的模型配置 YAML")
    parser.add_argument("--yes", action="store_true", help="跳过确认，直接应用")
    args = parser.parse_args()
    apply_best_params(args.tuning_dir, args.target, assume_yes=args.yes)


def apply_best_params(
    tuning_dir: str | Path,
    target_path: str | Path,
    assume_yes: bool = False,
    input_func: Callable[[str], str] = input,
) -> dict[str, Any]:
    """展示并可选应用最优参数。"""
    tuning_dir = Path(tuning_dir)
    target_path = Path(target_path)
    tuning_config = load_yaml_mapping(tuning_dir / "tuning_config.yaml")
    best_config = load_yaml_mapping(tuning_dir / "best_config.yaml")
    changes = collect_tuned_changes(target_path, tuning_config, best_config)
    print(format_changes(changes), flush=True)
    if not changes:
        return {"applied": False, "changes": changes}
    if not assume_yes:
        answer = input_func(f"Apply these changes to {target_path}? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Skipped.", flush=True)
            return {"applied": False, "changes": changes}
    raw_target = load_yaml_mapping(target_path)
    for change in changes:
        set_config_value(raw_target, change["path"], change["new"])
    save_yaml(raw_target, target_path)
    print(f"Applied {len(changes)} parameter change(s) to {target_path}.", flush=True)
    return {"applied": True, "changes": changes}


def collect_tuned_changes(target_path: str | Path, tuning_config: dict[str, Any], best_config: dict[str, Any]) -> list[dict[str, Any]]:
    """按 tuning search_space 收集目标配置与 best_config 的差异。"""
    target_config = load_config(target_path)
    changes = []
    for raw_name in tuning_config.get("search_space", {}):
        path = normalize_param_path(raw_name)
        old = get_config_value(target_config, path)
        new = get_config_value(best_config, path)
        if old != new:
            changes.append({"name": raw_name, "path": path, "old": old, "new": new})
    return changes


def get_config_value(config: dict[str, Any], dotted_path: str):
    target: Any = config
    for part in dotted_path.split("."):
        if not isinstance(target, dict) or part not in target:
            return None
        target = target[part]
    return target


def format_changes(changes: list[dict[str, Any]]) -> str:
    if not changes:
        return "No tuned parameter changes to apply."
    path_width = max(len(change["path"]) for change in changes)
    lines = [
        "",
        "Tuned Parameter Changes",
        "-" * max(48, path_width + 30),
        f"{'Parameter'.ljust(path_width)} | Current -> Tuned",
        "-" * max(48, path_width + 30),
    ]
    for change in changes:
        lines.append(f"{change['path'].ljust(path_width)} | {change['old']} -> {change['new']}")
    return "\n".join(lines)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must be a mapping: {path}")
    return payload


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
