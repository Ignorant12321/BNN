"""数据准备命令测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.prepare_data import load_processed_splits, prepare_data_from_config


def _write_raw_csvs(tmp_path: Path) -> tuple[Path, Path]:
    generation_path = tmp_path / "generation.csv"
    weather_path = tmp_path / "weather.csv"
    generation_rows = [
        "DATE_TIME,PLANT_ID,SOURCE_KEY,DC_POWER,AC_POWER,DAILY_YIELD,TOTAL_YIELD",
    ]
    weather_rows = [
        "DATE_TIME,PLANT_ID,SOURCE_KEY,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION",
    ]
    for i in range(20):
        ts = pd.Timestamp("2020-05-15 00:00:00") + pd.Timedelta(minutes=15 * i)
        generation_rows.append(f"{ts:%d-%m-%Y %H:%M},1,A,{i + 1},{i + 2},{i},{i}")
        weather_rows.append(f"{ts:%Y-%m-%d %H:%M:%S},1,W,25,30,{i / 10}")
    generation_path.write_text("\n".join(generation_rows) + "\n", encoding="utf-8")
    weather_path.write_text("\n".join(weather_rows) + "\n", encoding="utf-8")
    return generation_path, weather_path


def test_prepare_data_from_config_writes_processed_splits(tmp_path: Path):
    """数据准备应清洗原始 CSV，并按时间写出 train/val/test 文件。"""
    generation_path, weather_path = _write_raw_csvs(tmp_path)
    processed_dir = tmp_path / "processed"
    config = {
        "data": {
            "generation_path": str(generation_path),
            "weather_path": str(weather_path),
            "processed_dir": str(processed_dir),
            "fill_missing": True,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        }
    }

    result_dir = prepare_data_from_config(config)

    assert result_dir == processed_dir
    for name in ["train.csv", "val.csv", "test.csv", "split_info.json"]:
        assert (processed_dir / name).exists()
    train = pd.read_csv(processed_dir / "train.csv", parse_dates=["DATE_TIME"])
    val = pd.read_csv(processed_dir / "val.csv", parse_dates=["DATE_TIME"])
    test = pd.read_csv(processed_dir / "test.csv", parse_dates=["DATE_TIME"])
    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3
    assert train["DATE_TIME"].max() < val["DATE_TIME"].min() < test["DATE_TIME"].min()
    assert {"hour", "is_daylight", "last_ac_power"}.issubset(train.columns)
    split_info = json.loads((processed_dir / "split_info.json").read_text(encoding="utf-8"))
    assert split_info["train_rows"] == 14


def test_load_processed_splits_returns_none_when_files_missing(tmp_path: Path):
    """没有准备好的数据文件时，训练流程应能回退到原始 CSV 流程。"""
    config = {"data": {"processed_dir": str(tmp_path / "missing")}}

    assert load_processed_splits(config) is None


def test_load_processed_splits_reads_prepared_files(tmp_path: Path):
    """存在准备好的 train/val/test 文件时，应按时间戳类型读回。"""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for name, ts in [
        ("train.csv", "2020-05-15 00:00:00"),
        ("val.csv", "2020-05-16 00:00:00"),
        ("test.csv", "2020-05-17 00:00:00"),
    ]:
        (processed_dir / name).write_text(f"DATE_TIME,AC_POWER\n{ts},1.0\n", encoding="utf-8")

    splits = load_processed_splits({"data": {"processed_dir": str(processed_dir)}})

    assert splits is not None
    assert splits.train.loc[0, "DATE_TIME"] == pd.Timestamp("2020-05-15 00:00:00")
