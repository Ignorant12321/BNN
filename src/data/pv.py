"""光伏数据读取与滑动窗口构造。

功能：
    1. 读取原始发电数据和天气数据。
    2. 按时间聚合逆变器数据并合并天气数据。
    3. 从已切分或合并后的数据表构造模型需要的 history/weather/direct/target 窗口数组。

常用命令：
    python -m src.data.preprocess --config configs/data.yaml
    python -m src.data.split --config configs/data.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureColumns:
    """模型输入输出使用的列分组。"""

    history: list[str]
    weather: list[str]
    direct: list[str]
    target: str


@dataclass(frozen=True)
class WindowArrays:
    """一次监督预测数据集对应的 NumPy 数组。"""

    history: np.ndarray
    weather: np.ndarray
    direct: np.ndarray
    target: np.ndarray

    def as_batch(self, limit: int | None = None) -> dict[str, np.ndarray]:
        """把窗口数组转换成模型前向传播需要的 batch 字典。"""
        stop = limit if limit is not None else len(self.target)
        return {
            "history": self.history[:stop],
            "weather": self.weather[:stop],
            "direct": self.direct[:stop],
        }


DEFAULT_FEATURE_COLUMNS = FeatureColumns(
    history=["AC_POWER"],
    weather=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"],
    direct=["AC_POWER"],
    target="AC_POWER",
)


def load_plant_dataframe(generation_path: str | Path, weather_path: str | Path) -> pd.DataFrame:
    """读取并合并逆变器发电 CSV 与天气传感器 CSV。"""
    generation = pd.read_csv(generation_path)
    weather = pd.read_csv(weather_path)
    generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"], dayfirst=True)
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

    generation_agg = (
        generation.groupby("DATE_TIME", as_index=False)
        .agg(
            {
                "DC_POWER": "sum",
                "AC_POWER": "sum",
                "DAILY_YIELD": "sum",
                "TOTAL_YIELD": "sum",
            }
        )
        .sort_values("DATE_TIME")
    )
    weather_agg = (
        weather.groupby("DATE_TIME", as_index=False)
        .agg(
            {
                "AMBIENT_TEMPERATURE": "mean",
                "MODULE_TEMPERATURE": "mean",
                "IRRADIATION": "mean",
            }
        )
        .sort_values("DATE_TIME")
    )
    merged = generation_agg.merge(weather_agg, on="DATE_TIME", how="inner").sort_values("DATE_TIME")
    return merged.reset_index(drop=True)


def make_window_arrays(frame: pd.DataFrame, columns: FeatureColumns, lookback: int, horizon: int) -> WindowArrays:
    """从单个按时间排序的数据表构造监督学习滑动窗口。"""
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    required = set(columns.history + columns.weather + columns.direct + [columns.target])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    history_values = frame[columns.history].to_numpy(dtype=np.float32)
    weather_values = frame[columns.weather].to_numpy(dtype=np.float32)
    direct_values = frame[columns.direct].to_numpy(dtype=np.float32)
    target_values = frame[columns.target].to_numpy(dtype=np.float32)

    history_windows = []
    weather_windows = []
    direct_rows = []
    target_windows = []
    for start in range(0, len(frame) - lookback - horizon + 1):
        split = start + lookback
        end = split + horizon
        history_windows.append(history_values[start:split])
        weather_windows.append(weather_values[split:end])
        direct_rows.append(direct_values[split - 1])
        target_windows.append(target_values[split:end])

    if not target_windows:
        raise ValueError("not enough rows to build one forecasting window")
    return WindowArrays(
        history=np.stack(history_windows).astype(np.float32),
        weather=np.stack(weather_windows).astype(np.float32),
        direct=np.stack(direct_rows).astype(np.float32),
        target=np.stack(target_windows).astype(np.float32),
    )


def load_window_arrays_from_config(config: dict) -> WindowArrays:
    """根据原始 CSV 配置合并全量数据并构造窗口；正式训练不走此路径。"""
    data = config["data"]
    frame = load_plant_dataframe(data["generation_path"], data["weather_path"])
    return make_window_arrays_from_config(frame, config)


def load_split_window_arrays_from_config(config: dict) -> dict[str, WindowArrays]:
    """读取已切分的 train/val/test CSV，并在每个 split 内独立构造窗口，避免跨 split 泄露。"""
    data = config["data"]
    processed_dir = Path(data.get("processed_dir", "data/processed"))
    arrays_by_split = {}
    for split_name in ("train", "val", "test"):
        split_path = processed_dir / f"{split_name}.csv"
        frame = pd.read_csv(split_path)
        if "DATE_TIME" in frame.columns:
            frame["DATE_TIME"] = pd.to_datetime(frame["DATE_TIME"])
            frame = frame.sort_values("DATE_TIME").reset_index(drop=True)
        arrays_by_split[split_name] = make_window_arrays_from_config(frame, config)
    return arrays_by_split


def make_window_arrays_from_config(frame: pd.DataFrame, config: dict) -> WindowArrays:
    """根据配置中的特征列、lookback 和 horizon 从数据表构造窗口。"""
    data = config["data"]
    return make_window_arrays(
        frame,
        feature_columns_from_config(config),
        lookback=int(data["lookback"]),
        horizon=int(data["horizon"]),
    )


def feature_columns_from_config(config: dict) -> FeatureColumns:
    """从配置读取特征列；未配置时使用默认特征分组。"""
    features = config.get("data", {}).get("features")
    if not isinstance(features, dict):
        return DEFAULT_FEATURE_COLUMNS
    return FeatureColumns(
        history=list(features.get("history", DEFAULT_FEATURE_COLUMNS.history)),
        weather=list(features.get("weather", DEFAULT_FEATURE_COLUMNS.weather)),
        direct=list(features.get("direct", DEFAULT_FEATURE_COLUMNS.direct)),
        target=str(features.get("target", DEFAULT_FEATURE_COLUMNS.target)),
    )


def feature_dimensions_from_config(config: dict) -> tuple[int, int, int]:
    """根据 data.features 自动推断 history/weather/direct 输入维度。"""
    columns = feature_columns_from_config(config)
    return len(columns.history), len(columns.weather), len(columns.direct)
