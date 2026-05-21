"""训练产物元数据。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def build_manifest(
    *,
    run_dir: Path,
    config: dict[str, Any],
    started_at: datetime,
    ended_at: datetime,
    duration_seconds: float,
    split_sizes: dict[str, int],
    model_path: Path,
    best_epoch: int | None = None,
) -> dict[str, Any]:
    """构造可复现实验所需的最小 manifest。"""
    return {
        "schema_version": 1,
        "model_name": str(config.get("model", {}).get("name", "model")),
        "run_dir": str(run_dir),
        "model_file": str(model_path),
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "best_epoch": best_epoch,
        "split_sizes": split_sizes,
        "data": {
            "processed_dir": str(config.get("data", {}).get("processed_dir", "")),
            "lookback": config.get("data", {}).get("lookback"),
            "horizon": config.get("data", {}).get("horizon"),
        },
        "training": config.get("training", {}),
    }
