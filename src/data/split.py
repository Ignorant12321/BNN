"""数据集切分命令。

功能：
    读取 `processed_dir/plant_frame.csv`，按时间顺序切分为 train/val/test。

使用：
    python -m src.data.split --config configs/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="按时间顺序切分训练集、验证集和测试集。")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径")
    args = parser.parse_args()
    outputs = run_split(load_config(args.config))
    for split_name, path in outputs.items():
        print(f"{split_name}: {path}")


def run_split(config: dict) -> dict[str, Path]:
    """执行时间顺序切分，并返回三个输出文件路径。"""
    data = config["data"]
    processed_dir = Path(data.get("processed_dir", "data/processed"))
    input_path = processed_dir / "plant_frame.csv"
    frame = pd.read_csv(input_path)
    if "DATE_TIME" in frame.columns:
        frame["DATE_TIME"] = pd.to_datetime(frame["DATE_TIME"])
        frame = frame.sort_values("DATE_TIME").reset_index(drop=True)

    train_ratio = float(data.get("train_ratio", 0.7))
    val_ratio = float(data.get("val_ratio", 0.15))
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a non-empty test split")

    train_end = int(len(frame) * train_ratio)
    val_end = train_end + int(len(frame) * val_ratio)
    splits = {
        "train": frame.iloc[:train_end],
        "val": frame.iloc[train_end:val_end],
        "test": frame.iloc[val_end:],
    }
    outputs = {
        "train": processed_dir / "train.csv",
        "val": processed_dir / "val.csv",
        "test": processed_dir / "test.csv",
    }
    for split_name, split_frame in splits.items():
        split_frame.to_csv(outputs[split_name], index=False)
    return outputs


if __name__ == "__main__":
    main()
