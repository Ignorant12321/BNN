"""数据清洗与时间切分命令。

运行方式：

    python -m src.prepare_data --config configs/default.yaml

该命令把原始 CSV 清洗为电站级时间序列，添加基础特征，并按时间顺序
切分为 train/val/test 三个表格文件。训练入口会优先读取这些已处理文件；
如果文件不存在，则回退到原始 CSV 流程。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_plant_dataframe
from src.dataset import TimeSplits, build_time_splits
from src.features import add_basic_features
from src.utils import load_config, save_json


PROCESSED_FILENAMES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = prepare_data_from_config(config)
    print(f"Processed data written to: {output_dir}")


def prepare_data_from_config(config: dict) -> Path:
    """按配置清洗原始数据，并写出按时间划分后的数据表。"""
    data_config = config["data"]
    output_dir = processed_dir_from_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_plant_dataframe(
        data_config["generation_path"],
        data_config["weather_path"],
        fill_missing=data_config.get("fill_missing", True),
    )
    df = add_basic_features(df)
    splits = build_time_splits(df, data_config["train_ratio"], data_config["val_ratio"])
    _write_splits(splits, output_dir)
    save_json(_split_info(splits), output_dir / "split_info.json")
    return output_dir


def processed_dir_from_config(config: dict) -> Path:
    """返回已处理数据目录。"""
    return Path(config.get("data", {}).get("processed_dir", "data/processed"))


def load_processed_splits(config: dict) -> TimeSplits | None:
    """如果已处理 train/val/test 文件存在，则读回时间切分结果。"""
    processed_dir = processed_dir_from_config(config)
    paths = {name: processed_dir / filename for name, filename in PROCESSED_FILENAMES.items()}
    if not all(path.exists() for path in paths.values()):
        return None
    return TimeSplits(
        train=_read_processed_csv(paths["train"]),
        val=_read_processed_csv(paths["val"]),
        test=_read_processed_csv(paths["test"]),
    )


def _write_splits(splits: TimeSplits, output_dir: Path) -> None:
    for name, filename in PROCESSED_FILENAMES.items():
        split = getattr(splits, name)
        split.to_csv(output_dir / filename, index=False)


def _read_processed_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["DATE_TIME"])


def _split_info(splits: TimeSplits) -> dict[str, str | int]:
    return {
        "train_start": str(splits.train["DATE_TIME"].min()),
        "train_end": str(splits.train["DATE_TIME"].max()),
        "train_rows": int(len(splits.train)),
        "val_start": str(splits.val["DATE_TIME"].min()),
        "val_end": str(splits.val["DATE_TIME"].max()),
        "val_rows": int(len(splits.val)),
        "test_start": str(splits.test["DATE_TIME"].min()),
        "test_end": str(splits.test["DATE_TIME"].max()),
        "test_rows": int(len(splits.test)),
    }


if __name__ == "__main__":
    main()
