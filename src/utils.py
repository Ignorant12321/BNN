"""通用工具函数。

本文件集中放置配置读写、随机种子、设备选择、实验目录创建和对象保存。
训练脚本需要频繁使用这些能力，单独拆出来可以让 `train.py` 保持清晰。
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """保存 YAML 配置，便于每次实验结果可复现。"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。

    cuDNN 设置为 deterministic 可以提高复现性，但可能略微降低训练速度。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ModuleNotFoundError:
        pass


def resolve_device(device_name: str):
    """解析训练设备。

    配置为 `auto` 时优先使用 CUDA；显式请求 cuda 但不可用时给出警告并回退 CPU。
    """
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def describe_device(device, cuda_available: bool | None = None, cuda_name: str | None = None) -> str:
    """返回训练设备状态，便于确认是否真的使用 GPU。"""
    if cuda_available is None or (cuda_available and cuda_name is None):
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available and cuda_name is None:
            cuda_name = torch.cuda.get_device_name(0)
    parts = [f"device={device}", f"cuda_available={cuda_available}"]
    if cuda_name:
        parts.append(f"gpu={cuda_name}")
    return ", ".join(parts)


def create_run_dir(base_dir: str | Path, model_name: str, note: str | None = None) -> Path:
    """创建一次实验的输出目录。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / model_name / timestamp
    for child in ["checkpoints", "metrics", "figures", "predictions", "logs", "artifacts"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    (run_dir / "note.txt").write_text(note or timestamp, encoding="utf-8")
    return run_dir


def setup_logger(log_path: str | Path) -> logging.Logger:
    """创建同时输出到控制台和日志文件的 logger。"""
    logger = logging.getLogger("pv_forecasting")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json(data: Any, path: str | Path) -> None:
    """保存 JSON 文件，保留中文字符。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_pickle(obj: Any, path: str | Path) -> None:
    """保存 Python 对象，例如 scaler。"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """读取 pickle 对象。"""
    with open(path, "rb") as f:
        return pickle.load(f)
