"""训练和对比产物的文件系统布局。"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.models.registry import build_model
from src.torch_runtime import import_torch


def create_run_dir(config: dict[str, Any]) -> Path:
    """创建训练产物目录；默认 outputs/train/<model>/<timestamp>。"""
    explicit_run_dir = config.get("run_dir")
    if explicit_run_dir:
        run_dir = Path(str(explicit_run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    model_name = str(config.get("model", {}).get("name", "model"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.get("output_dir", "outputs")) / "train" / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_comparison_dir(name: str, output_dir: str | Path = "outputs") -> Path:
    """创建 outputs/comparisons/<timestamp>。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    compare_dir = Path(output_dir) / "comparisons" / timestamp
    compare_dir.mkdir(parents=True, exist_ok=True)
    return compare_dir


def save_config(config: dict[str, Any], path: Path) -> None:
    """保存配置快照。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_run_note(run_dir: Path, note: str | None) -> Path:
    """写出产物备注；没有备注时使用产物目录名。"""
    note_text = run_dir.name if note is None else note
    note_path = run_dir / "note.txt"
    note_path.write_text(f"{note_text}\n", encoding="utf-8")
    return note_path


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    """读取训练目录中保存的配置快照。"""
    config_path = Path(run_dir) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"training config must be a mapping: {config_path}")
    return payload


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """保存 manifest.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def save_model_artifact(model, config: dict[str, Any], models_dir: Path, stem: str = "best") -> Path:
    """保存模型文件，Torch 使用 .pt，NumPy/对象模型使用 .pkl。"""
    models_dir.mkdir(parents=True, exist_ok=True)
    if getattr(model, "is_torch_model", False):
        path = models_dir / f"{stem}.pt"
        torch = import_torch()
        torch.save(
            {
                "model_class": type(model).__name__,
                "model_state_dict": model.state_dict(),
                "config": config,
            },
            path,
        )
        return path

    path = models_dir / f"{stem}.pkl"
    payload: dict[str, Any] = {"model": model, "model_class": type(model).__name__, "config": config}
    try:
        with path.open("wb") as file:
            pickle.dump(payload, file)
    except Exception as error:
        with path.open("wb") as file:
            pickle.dump(
                {
                    "model_class": type(model).__name__,
                    "config": config,
                    "pickle_error": str(error),
                },
                file,
            )
    return path


def load_model_artifact(run_dir: str | Path):
    """从训练目录加载 best 模型。"""
    run_path = Path(run_dir)
    config = load_run_config(run_path)
    torch_path = run_path / "models" / "best.pt"
    if torch_path.is_file():
        torch = import_torch()
        payload = torch.load(torch_path, map_location="cpu")
        model = build_model(config)
        model.load_state_dict(payload["model_state_dict"])
        return model, config

    pickle_path = run_path / "models" / "best.pkl"
    if pickle_path.is_file():
        with pickle_path.open("rb") as file:
            payload = pickle.load(file)
        model = payload.get("model")
        if model is None:
            raise ValueError(f"pickle payload does not contain a loadable model: {pickle_path}")
        return model, config

    raise FileNotFoundError(f"model artifact not found in: {run_path / 'models'}")


def resolve_run_path(path: str | Path) -> Path:
    """解析训练产物路径；传入模型根目录时选择最新时间戳目录。"""
    run_path = Path(path)
    if is_run_dir(run_path):
        return run_path
    if not run_path.is_dir():
        return run_path
    candidates = [child for child in run_path.iterdir() if child.is_dir() and is_run_dir(child)]
    if not candidates:
        return run_path
    return max(candidates, key=lambda candidate: candidate.name)


def is_run_dir(path: Path) -> bool:
    """判断路径是否像一个训练目录。"""
    return (
        (path / "config.yaml").is_file()
        or (path / "metrics.csv").is_file()
        or (path / "epoch_history.csv").is_file()
    )
