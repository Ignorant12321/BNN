"""配置文件读取工具。

功能：
    读取 YAML 配置，并保证顶层结构是字典。

    使用：
    该文件通常被其他命令导入，例如：
    python -m src.experiments.train --config configs/models/bnn/24h.yaml
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件，处理 include，并补齐默认训练配置。"""
    config_path = Path(path)
    config = _load_config_file(config_path)
    return apply_config_defaults(config)


def _load_config_file(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError("top-level YAML object must be a mapping")
    merged: dict[str, Any] = {}
    includes = payload.pop("include", [])
    if isinstance(includes, (str, Path)):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError("include must be a string or list of strings")
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = config_path.parent / include_path
        deep_update(merged, _load_config_file(include_path))
    deep_update(merged, payload)
    return merged


def apply_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """补齐项目默认配置，减少每个模型 yaml 的重复内容。"""
    result = copy.deepcopy(config)
    result.setdefault("seed", 42)
    result.setdefault("output_dir", "outputs")
    training = result.setdefault("training", {})
    training.setdefault("backend", "torch")
    training.setdefault("device", "auto")
    training.setdefault("epochs", 5)
    training.setdefault("batch_size", 32)
    training.setdefault("lr", 0.001)
    training.setdefault("weight_decay", 0.0)
    training.setdefault("kl_beta", 0.0)
    return result


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置字典。"""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target
